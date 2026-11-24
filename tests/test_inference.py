"""Tests: model inference path (offline == streaming)."""
import numpy as np
import pandas as pd
import pytest

from src.models.inference import SepsiGuardPredictor
from src.training.data import TorchWindows, WindowData, predict_logits


@pytest.fixture(scope="module")
def predictor():
    return SepsiGuardPredictor()


@pytest.fixture(scope="module")
def test_data():
    return WindowData("data/processed/temporal_windows/test.npz")


def test_registry_manifest_complete(predictor):
    m = predictor.manifest
    assert len(m["feature_order"]) == 69
    assert m["input_spec"]["n_channels"] == 69
    assert m["input_spec"]["lookback_hours"] == 12
    for h in ("y6", "y12"):
        assert "a" in m["platt"][h] and "b" in m["platt"][h]
        assert "alert" in m["thresholds"][h]


def test_pending_history_reported_not_zero(predictor):
    hist = pd.DataFrame({"iculos": [0, 1, 2], "hr": [80.0, 81.0, 79.0]})
    out = predictor.predict_history(hist)
    assert out is None or out.get("prediction_status") != "ok"
    res = predictor.features_from_history(hist)
    assert res is None  # incomplete 12 h history -> no prediction


def test_streaming_matches_offline_batch(predictor, test_data):
    """The streaming predictor (raw history -> predict_history) must
    reproduce the offline batch inference over the stored windows for a
    handful of real test patients."""
    import torch
    from src.features.config import RAW_TO_FEATURE
    obs = pd.read_parquet("data/interim/validated/observations.parquet")
    rng = np.random.default_rng(0)
    pids = rng.choice(np.unique(test_data.pid), size=5, replace=False)
    for pid in pids:
        idx = np.flatnonzero(test_data.pid == pid)[5]
        i = int(idx)
        # rebuild raw history up to t
        t = int(test_data.t[i])
        hist = obs[(obs.patient_id == pid) & (obs.ICULOS <= t)].rename(
            columns={**RAW_TO_FEATURE, "ICULOS": "iculos"})
        cols = ["iculos"] + list(RAW_TO_FEATURE.values())
        cols = [c for c in cols if c != "iculos"] 
        hist = hist[["iculos"] + cols].sort_values("iculos")
        out = predictor.predict_history(hist)
        assert out is not None, f"patient {pid} hour {t} should have full history"
        # compare with offline: apply platt to stored batch logits
        import json
        man = json.loads(open("artifacts/model_registry/sepsiguard_v1.0/manifest.json").read())
        a12, b12 = man["platt"]["y12"]["a"], man["platt"]["y12"]["b"]
        from scipy.special import expit
        ref_logit = _offline_logit(predictor, test_data, i)
        p_ref = expit(a12 * ref_logit + b12)
        assert abs(out["prob_12h"] - p_ref) < 2e-4, (out["prob_12h"], p_ref)


def _offline_logit(predictor, data, i):
    """Run the model on stored window i and return the 12 h logit."""
    import torch
    x = torch.from_numpy(data.X[i])[None]
    mask = x[:, :, 34:68]
    delta = torch.from_numpy(data.D[i])[None]
    with torch.no_grad():
        _, l12 = predictor.model(x, mask, delta)
    return float(l12[0])


def test_risk_bands():
    assert SepsiGuardPredictor.risk_band(0.01) == "low"
    assert SepsiGuardPredictor.risk_band(0.07) == "elevated"
    assert SepsiGuardPredictor.risk_band(0.3) == "high"
    assert SepsiGuardPredictor.risk_band(0.7) == "critical"