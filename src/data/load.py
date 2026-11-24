"""Raw PhysioNet 2019 ICU dataset loading and caching.

The challenge distribution stores one pipe-separated file per patient with
hourly observations. Columns (official schema, Reyna et al. 2020):

    34 clinical channels, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS, SepsisLabel

This module concatenates the per-patient files into a single validated table
(`data/interim/validated/observations.parquet`) so that downstream stages do
not re-read 40,336 individual files.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "physionet_2019"
CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "interim" / "validated" / "observations.parquet"
)

CLINICAL_CHANNELS = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine",
    "Bilirubin_direct", "Glucose", "Lactate", "Magnesium",
    "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
    "PTT", "WBC", "Fibrinogen", "Platelets",
]
DEMOGRAPHIC_CHANNELS = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime"]
TIME_CHANNELS = ["ICULOS"]
LABEL_CHANNEL = ["SepsisLabel"]
ALL_COLUMNS = CLINICAL_CHANNELS + DEMOGRAPHIC_CHANNELS + TIME_CHANNELS + LABEL_CHANNEL


def _parse_one(args):
    """Read a single patient PSV file -> DataFrame with patient_id and dataset."""
    path, dataset = args
    pid = int(Path(path).stem[1:])  # 'p000001' -> 1 ; 'p100001' -> 100001
    df = pd.read_csv(path, sep="|", dtype=float)
    df["patient_id"] = pid
    df["dataset"] = dataset
    return df


def _scan_set(dataset: str) -> pd.DataFrame:
    """Concatenate all patient files of one training set into one frame."""
    folder = RAW_DIR / dataset
    frames = []
    paths = sorted(folder.glob("*.psv"))
    if not paths:
        raise FileNotFoundError(f"No PSV files under {folder}. See README for dataset placement.")
    with ProcessPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as ex:
        for df in ex.map(_parse_one, [(p, dataset) for p in paths]):
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_cache() -> pd.DataFrame:
    """Build the validated observation cache (one row per patient-hour)."""
    df = pd.concat([_scan_set("training_setA"), _scan_set("training_setB")], ignore_index=True)
    df = df.sort_values(["patient_id", "ICULOS"], kind="stable").reset_index(drop=True)
    # ---- structural validation --------------------------------------------
    missing_cols = set(ALL_COLUMNS) - set(df.columns)
    assert not missing_cols, f"Schema violation: missing {missing_cols}"
    assert (df["ICULOS"] >= 0).all(), "ICULOS must be non-negative"
    monotonic = df.groupby("patient_id")["ICULOS"].apply(lambda s: s.is_monotonic_increasing)
    assert monotonic.all(), "non-chronological patient file"
    dup = df.duplicated(subset=["patient_id", "ICULOS"])
    if dup.any():
        df = df.drop_duplicates(subset=["patient_id", "ICULOS"], keep="first")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PATH, index=False)
    return df


def load_observations(use_cache: bool = True) -> pd.DataFrame:
    """Return the full validated observation table (builds cache if needed)."""
    if use_cache and CACHE_PATH.exists():
        return pd.read_parquet(CACHE_PATH)
    return build_cache()


def sepsis_onset_hour(patient: pd.DataFrame) -> float:
    """Operational onset hour: first ICULOS with SepsisLabel == 1, else NaN.

    A single positive hour already defines sepsis onset under the
    challenge's clinical presumption (SepsisLabel is derived from the
    Sepsis-3 timewindow definition used by the challenge organisers)."""
    pos = patient.loc[patient["SepsisLabel"] == 1, "ICULOS"]
    return float(pos.min()) if len(pos) else float("nan")


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with onset hour and stay length."""
    rows = []
    for pid, g in df.groupby("patient_id", sort=True):
        onset = sepsis_onset_hour(g)
        rows.append(
            {
                "patient_id": pid,
                "dataset": g["dataset"].iloc[0],
                "n_hours": len(g),
                "stay_start": float(g["ICULOS"].min()),
                "stay_end": float(g["ICULOS"].max()),
                "age": g["Age"].iloc[-1],
                "gender": g["Gender"].iloc[-1],
                "hosp_adm_time": g["HospAdmTime"].iloc[-1],
                "septic": 0 if np.isnan(onset) else 1,
                "onset_hour": onset,
            }
        )
    return pd.DataFrame(rows)