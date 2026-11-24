"""Standalone real-time inference for SepsiGuard.

This module is the single inference path shared by historical ICU replay
(Phase 9) and any authorised prospective integration. It depends ONLY on:
  * the model registry artifacts (weights + manifest),
  * raw observation rows,
and never on training-time state.

Prediction contract:
  * A prediction is produced for every ingested hour that completes a
    12-hour history for the patient.
  * If history is incomplete the prediction status is 'pending_history'
    and NO risk probability is reported (never 0 risk).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from src.features.build import preprocess_patient
from src.features.config import CLINICAL_FEATURES, FEATURE_ORDER, LOOKBACK
from src.models.sepsiguard import SepsiGuardNet as SepsiGuardNet

REGISTRY = Path(__file__).resolve().parents[2] / "artifacts" / "model_registry"


class SepsiGuardPredictor:
    """Loads a registry entry and performs per-patient streaming inference."""

    def __init__(self, version: str = "sepsiguard_v1.0", registry: Path = REGISTRY):
        self.dir = registry / version
        manifest = json.loads((self.dir / "manifest.json").read_text())
        self.manifest = manifest
        self.normalizer = manifest["normalizer"]
        ck = torch.load(self.dir / "model.pt", map_location="cpu", weights_only=False)
        self.model = SepsiGuardNet(
            use_gate=ck["use_gate"], dropout=ck.get("dropout", 0.2))
        self.model.load_state_dict(ck["state_dict"])
        self.model.eval()
        self.platt = manifest["platt"]          # {"y6": {a,b}, "y12": {a,b}}
        self.thresholds = manifest["thresholds"]
        self.version = version

    # ------------------------------------------------------------------ state
    def features_from_history(self, history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        """history: raw observations (columns = 34 clinical + iculos),
        chronological. Returns (X (12,69), D (12,34)) for the last hour,
        or None when the 12 h history is incomplete."""
        if len(history) < LOOKBACK:
            return None
        g = history.reset_index(drop=True)
        values, mask, delta = preprocess_patient(g, self.normalizer)
        icu = g["iculos"].to_numpy(dtype=np.float32)
        icu_norm = (icu - self.normalizer["iculos_mean"]) / (self.normalizer["iculos_std"] or 1.0)
        X_full = np.concatenate([values, mask, icu_norm[:, None]], axis=1).astype(np.float32)
        X = X_full[-LOOKBACK:]            # last 12 preprocessed rows
        D = delta[-LOOKBACK:]
        return X, D

    # ---------------------------------------------------------------- predict
    def predict_history(self, history: pd.DataFrame) -> dict | None:
        """Full prediction for the last hour of `history` (raw rows)."""
        out = self.features_from_history(history)
        if out is None:
            return None
        X, D = out
        x = torch.from_numpy(X)[None]                    # (1, 12, 69)
        mask = x[:, :, 34:68]
        delta = torch.from_numpy(D)[None]                # (1, 12, 34)
        with torch.no_grad():
            l6, l12 = self.model(x, mask, delta)
        l6, l12 = float(l6[0]), float(l12[0])
        p6 = float(expit(self.platt["y6"]["a"] * l6 + self.platt["y6"]["b"]))
        p12 = float(expit(self.platt["y12"]["a"] * l12 + self.platt["y12"]["b"]))
        return {
            "prob_6h": p6, "prob_12h": p12,
            "logit_6h": l6, "logit_12h": l12,
            "alert_6h": bool(p6 >= self.thresholds["y6"]["alert"]),
            "alert_12h": bool(p12 >= self.thresholds["y12"]["alert"]),
            "model_version": self.version,
        }

    # ------------------------------------------------------------- risk band
    @staticmethod
    def risk_band(p12: float) -> str:
        if p12 >= 0.5:
            return "critical"
        if p12 >= 0.2:
            return "high"
        if p12 >= 0.05:
            return "elevated"
        return "low"