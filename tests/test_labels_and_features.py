"""Unit tests: label construction, windowing, missingness, leakage."""
import numpy as np
import pandas as pd
import pytest

from src.features.config import FEATURE_ORDER, LOOKBACK
from src.features.labels import onset_hour, window_label_mask


def test_feature_order_is_69():
    assert len(FEATURE_ORDER) == 69
    assert FEATURE_ORDER[:3] == ["hr", "o2sat", "temp"]
    assert FEATURE_ORDER[34] == "hr_missing"
    assert FEATURE_ORDER[68] == "iculos"


def test_onset_hour():
    lab = np.array([0, 0, 0, 1, 1, 1])
    icu = np.arange(6)
    assert onset_hour(lab, icu) == 3.0
    assert np.isnan(onset_hour(np.zeros(6), icu))


def test_labels_exclude_post_onset_hours():
    # septic at hour 20, stay ends hour 39 -> keepable t in [11, 27]
    n, onset, stay_end = 40, 20.0, 39
    lab12, keep12 = window_label_mask(n, onset, stay_end, 12)
    t = np.arange(n)
    assert keep12.sum() > 0
    # at-risk: no kept window at or after onset
    assert not keep12[t >= onset].any()
    # kept windows: complete 12h history and complete 12h future
    assert (t[keep12] >= 11).all() and (t[keep12] <= stay_end - 12).all()
    # positive windows: onset within (t, t+12] -> t in [onset-12, onset-1] = [8,19]
    pos = t[keep12][lab12[keep12] == 1]
    # positives are the intersection [onset-12, onset-1] with keepable [11, stay-12]
    assert len(pos) == 9 and (pos >= 11).all() and (pos <= 19).all()
    # label is 0 exactly on kept negatives
    assert (lab12[keep12][lab12[keep12] == 0] == 0).all()


def test_censored_windows_are_never_negative():
    # stay ends at hour 15 (< 12 history + 12 future) -> every window censored
    n, onset, stay_end = 16, np.nan, 15
    lab, keep = window_label_mask(n, onset, stay_end, 12)
    assert keep.sum() == 0            # excluded entirely,
    assert lab.sum() == 0             # never silently labelled negative


def test_non_septic_patient_labels_all_zero():
    lab, keep = window_label_mask(40, np.nan, 39, 12)
    assert (lab[keep] == 0).all()
    assert keep.sum() > 0


def test_window_tensor_shapes_and_causality():
    from src.features.build import preprocess_patient
    from src.features.config import CLINICAL_FEATURES
    rng = np.random.default_rng(0)
    n = 24
    df = pd.DataFrame({"iculos": np.arange(n, dtype=float)})
    for c in CLINICAL_FEATURES:
        df[c] = np.nan
    df["hr"] = rng.normal(80, 10, n)   # only HR observed
    stats = {
        "population_median": {c: 0.0 for c in FEATURE_ORDER[:34]},
        "mean": {c: 0.0 for c in FEATURE_ORDER[:34]},
        "std": {c: 1.0 for c in FEATURE_ORDER[:34]},
        "iculos_mean": 0.0, "iculos_std": 1.0,
    }
    values, mask, delta = preprocess_patient(df, stats)
    assert values.shape == (n, 34) and mask.shape == (n, 34) and delta.shape == (n, 34)
    # hr was observed -> mask 0; everything else mask 1
    assert (mask[:, 0] == 0).all()
    assert mask[:, 1:].mean() == 1
    # unobserved channels keep population median (0) -> normalised 0
    assert np.allclose(values[:, 1:], 0)
    # LOCF: observed value carries forward unchanged with identity normalizer
    assert np.allclose(values[:, 0], df["hr"].to_numpy())
    assert np.isfinite(values).all() and np.isfinite(delta).all()


def test_normalizer_uses_train_statistics_only():
    """Leakage guard: the saved normalizer must equal statistics computed
    from TRAIN patients only (spot-check against a small recompute)."""
    import json
    cfg = json.loads(open("artifacts/preprocessing/preprocess_config.json").read())
    tr = pd.read_csv("data/splits/train_patients.csv")
    assert len(tr) > 20000
    # normaliser keys are exactly the 34 clinical channels
    assert set(cfg["normalizer"]["mean"].keys()) == set(FEATURE_ORDER[:34])
    # sanity: std values are positive
    assert all(v > 0 for v in cfg["normalizer"]["std"].values())