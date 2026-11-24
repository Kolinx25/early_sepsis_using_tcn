"""Phase 2 — Reproducible preprocessing and temporal window construction.

Pipeline (all parameters fitted on TRAIN patients only):

1. Chronological ordering, duplicate (patient, ICULOS) removal (loader).
2. Raw -> feature tensor per patient: 34 clinical channels.
3. Missingness indicators (1 = absent in raw record).
4. Delta channels: hours since last observation, capped at 48, scaled /48.
5. Imputation: forward-fill within patient (LOCF, clinically justified for
   ICU monitoring data, cf. Che et al. 2018), then population median
   (train-only) for leading gaps.
6. Normalisation: z-score using train-only mean/std of imputed values;
   ICULOS scaled by train mean/std as channel 68.
7. Windows: sliding 12 h history X(t) = [t-11 .. t], labels Y6/Y12 per
   src/features/labels.py.

Outputs per split (data/processed/temporal_windows/):
    X.npy     float32 (N, 12, 69)   model input
    D.npy     float32 (N, 12, 34)   delta (gate branch input)
    y6.npy    uint8
    y12.npy   uint8
    pid.npy   int64   patient ids
    t.npy     int32   prediction hour (ICULOS)
Artifacts: artifacts/preprocessing/preprocess_config.json,
           artifacts/scalers/normalizer.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.load import load_observations
from src.features.config import (
    CLINICAL_FEATURES, DELTA_CAP_HOURS, FEATURE_ORDER, HORIZONS, LOOKBACK,
    RAW_TO_FEATURE, SEED, SPLIT_FRACS,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "temporal_windows"
SPLIT_DIR = ROOT / "data" / "splits"
ART_PRE = ROOT / "artifacts" / "preprocessing"
ART_SCALE = ROOT / "artifacts" / "scalers"


def rename_to_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=RAW_TO_FEATURE)
    df = df.rename(columns={"ICULOS": "iculos", "SepsisLabel": "sepsis_label"})
    return df


def fit_train_statistics(df: pd.DataFrame, train_pids: np.ndarray) -> dict:
    """Population medians + z-score parameters, TRAIN patients only."""
    tr = df[df.patient_id.isin(train_pids)]
    med = tr[CLINICAL_FEATURES].median(numeric_only=True)
    # After LOCF imputation on train patients:
    filled = tr.groupby("patient_id")[CLINICAL_FEATURES].ffill()
    # fill remaining leading gaps with the (train) median
    filled = filled.fillna(med)
    mean = filled.mean()
    std = filled.std().replace(0, 1.0)
    iculos_mean = float(tr["iculos"].mean())
    iculos_std = float(tr["iculos"].std() or 1.0)
    return {
        "population_median": {c: float(med[c]) for c in CLINICAL_FEATURES},
        "mean": {c: float(mean[c]) for c in CLINICAL_FEATURES},
        "std": {c: float(std[c]) for c in CLINICAL_FEATURES},
        "iculos_mean": iculos_mean,
        "iculos_std": iculos_std,
    }


def preprocess_patient(
    g: pd.DataFrame, stats: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (imputed+normalised values (n,34), mask (n,34), delta (n,34))."""
    n = len(g)
    raw = g[CLINICAL_FEATURES].to_numpy(dtype=np.float32)  # may contain NaN
    mask = np.isnan(raw).astype(np.float32)
    # ---- LOCF imputation -----------------------------------------------------
    # forward-fill within the patient's stay; leading gaps get the
    # train-population median (a patient-level constant imputation)
    med = np.array([stats["population_median"][c] for c in CLINICAL_FEATURES],
                   dtype=np.float32)
    filled = pd.DataFrame(raw).ffill().to_numpy(dtype=np.float32).copy()
    # leading gaps (before the patient's first observation) receive the
    # train-population median (patient-level constant imputation)
    first_obs = np.array([np.argmax(~np.isnan(raw[:, j])) if (~np.isnan(raw[:, j])).any()
                          else n for j in range(raw.shape[1])], dtype=int)
    for j in range(raw.shape[1]):
        if first_obs[j] > 0:
            filled[: first_obs[j], j] = med[j]
    # ---- delta: hours since last observation --------------------------------
    delta = np.full_like(raw, DELTA_CAP_HOURS)
    hours = g["iculos"].to_numpy(dtype=np.float32)
    for j in range(raw.shape[1]):
        if (~np.isnan(raw[:, j])).sum() == 0:
            continue  # channel never observed for this patient -> stays at cap
        # index of last observation at or before each hour
        last = np.maximum.accumulate(np.where(~np.isnan(raw[:, j]), np.arange(n), -1))
        last_hour = np.where(last >= 0, hours[np.clip(last, 0, None)],
                             hours[0] - DELTA_CAP_HOURS)
        delta[:, j] = np.clip(hours - last_hour, 0, DELTA_CAP_HOURS)
    delta = delta / DELTA_CAP_HOURS
    # ---- normalisation (train-only stats) ------------------------------------
    mu = np.array([stats["mean"][c] for c in CLINICAL_FEATURES], dtype=np.float32)
    sd = np.array([stats["std"][c] for c in CLINICAL_FEATURES], dtype=np.float32)
    values = (filled - mu) / sd
    return values.astype(np.float32), mask, delta.astype(np.float32)


def build_all(save: bool = True) -> dict:
    df = rename_to_features(load_observations())
    # ---- eligibility & onset per patient --------------------------------------
    info = []
    for pid, g in df.groupby("patient_id", sort=True):
        lab = g["sepsis_label"].to_numpy()
        icu = g["iculos"].to_numpy()
        onset = float(icu[np.flatnonzero(lab == 1)[0]]) if (lab == 1).any() else np.nan
        info.append((pid, g["dataset"].iloc[0], len(g), int(icu.max()), onset))
    info = pd.DataFrame(info, columns=["patient_id", "dataset", "n_hours",
                                       "stay_end", "onset"])
    # A stay must cover a full 12 h history AND a full 12 h future horizon
    # for at least one prediction window -> minimum stay length 24 h.
    eligible = info[info.n_hours >= LOOKBACK + HORIZONS["y12"]]

    # ---- patient-level stratified split ---------------------------------------
    el = eligible.copy()
    el["septic"] = np.where(np.isnan(el.onset), 0, 1)
    tr_v, test = train_test_split(
        el, test_size=SPLIT_FRACS["test"], stratify=el.septic, random_state=SEED)
    train, val = train_test_split(
        tr_v, test_size=SPLIT_FRACS["validation"] / (1 - SPLIT_FRACS["test"]),
        stratify=tr_v.septic, random_state=SEED)

    splits = {"train": train, "validation": val, "test": test}
    if save:
        SPLIT_DIR.mkdir(parents=True, exist_ok=True)
        for name, s in splits.items():
            s[["patient_id", "dataset", "septic", "onset", "n_hours"]].to_csv(
                SPLIT_DIR / f"{name}_patients.csv", index=False)

    stats = fit_train_statistics(df, train.patient_id.to_numpy())

    out: dict[str, dict] = {}
    for name, s in splits.items():
        pid_set = set(s.patient_id)
        Xs, Ds, y6s, y12s, pids, ts = [], [], [], [], [], []
        for pid, g in df.groupby("patient_id", sort=True):
            if pid not in pid_set:
                continue
            g = g.reset_index(drop=True)
            values, mask, delta = preprocess_patient(g, stats)
            n = len(g)
            icu = g["iculos"].to_numpy(dtype=np.float32)
            labels = g["sepsis_label"].to_numpy()
            onset = float(icu[np.flatnonzero(labels == 1)[0]]) if (labels == 1).any() else np.nan
            stay_end = int(icu.max())
            from src.features.labels import window_label_mask
            lab12, keep = window_label_mask(n, onset, stay_end, HORIZONS["y12"])
            lab6, _ = window_label_mask(n, onset, stay_end, HORIZONS["y6"])
            # note: keep uses the 12 h horizon (strictest censoring rule);
            # Y6 is evaluable on every kept window by construction.
            idx = np.flatnonzero(keep)
            if len(idx) == 0:
                continue
            # assemble per-hour 69-channel input rows
            icu_norm = (icu - stats["iculos_mean"]) / (stats["iculos_std"] or 1.0)
            X = np.concatenate(
                [values, mask, icu_norm[:, None].astype(np.float32)], axis=1
            ).astype(np.float32)  # (n, 69)
            # sliding windows via stride tricks: window index i covers hours
            # [i, i+11]; prediction hour t corresponds to window index t-11.
            from numpy.lib.stride_tricks import sliding_window_view
            widx = idx - (LOOKBACK - 1)
            winX = sliding_window_view(X, LOOKBACK, axis=0)[widx]      # (k, 69, 12)
            winD = sliding_window_view(delta, LOOKBACK, axis=0)[widx]  # (k, 34, 12)
            Xs.append(winX.transpose(0, 2, 1))   # (k, 12, 69)
            Ds.append(winD.transpose(0, 2, 1))   # (k, 12, 34)
            y6s.append(lab6[idx].astype(np.uint8))
            y12s.append(lab12[idx].astype(np.uint8))
            pids.append(np.full(len(idx), pid, dtype=np.int64))
            ts.append(icu[idx].astype(np.int32))
        out[name] = {
            "X": np.concatenate(Xs), "D": np.concatenate(Ds),
            "y6": np.concatenate(y6s), "y12": np.concatenate(y12s),
            "pid": np.concatenate(pids), "t": np.concatenate(ts),
        }
        print(f"[{name}] windows={len(out[name]['y12']):,} "
              f"patients={len(np.unique(out[name]['pid'])):,} "
              f"y6+={out[name]['y6'].sum():,} y12+={out[name]['y12'].sum():,}")

    if save:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, o in out.items():
            np.savez(OUT_DIR / f"{name}.npz",
                     X=o["X"], D=o["D"], y6=o["y6"], y12=o["y12"],
                     pid=o["pid"], t=o["t"])
        manifest = {
            "feature_order": FEATURE_ORDER,
            "lookback_hours": LOOKBACK,
            "horizons": HORIZONS,
            "delta_cap_hours": DELTA_CAP_HOURS,
            "imputation": "LOCF within patient + train-population median for leading gaps",
            "normalisation": "z-score on train-patient imputed values; ICULOS z-scored",
            "censoring_policy": "windows require complete 12h future horizon; excluded, never negative",
            "at_risk_policy": "prediction hours strictly before onset",
            "seed": SEED,
            "normalizer": stats,
        }
        ART_PRE.mkdir(parents=True, exist_ok=True)
        (ART_PRE / "preprocess_config.json").write_text(json.dumps(manifest, indent=2))
        ART_SCALE.mkdir(parents=True, exist_ok=True)
        (ART_SCALE / "normalizer.json").write_text(json.dumps(
            {"normalizer": stats, "feature_order": FEATURE_ORDER}, indent=2))
    return out


if __name__ == "__main__":
    build_all()