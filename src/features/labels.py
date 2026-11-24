"""Phase 3 — Sepsis prediction label construction.

Operational definitions (documented in docs/data_dictionary.md):

* Onset hour `t_onset`  : the first ICULOS hour at which SepsisLabel == 1
  within a patient's ICU stay. Consistent with the PhysioNet/CinC 2019
  challenge convention (Reyna et al. 2020), where SepsisLabel remains 1 for
  the remainder of the stay after onset. We verified empirically that
  99.98% of positive hours occur in multi-hour runs (median run 10 h).

* At-risk population    : hours t with 0 <= t < t_onset (strictly before
  onset). Post-onset hours are NOT prediction targets - the clinical
  question is advance warning, not retrospective labelling.

* Label Y_H(t)          : 1 if onset occurs in (t, t+H], else 0.
  Requires a COMPLETE future horizon of H hours within the stay
  (t + H <= stay_end). Windows with incomplete future horizon are
  EXCLUDED (censored), never labelled negative.

* Excluded observations  : (a) t < 11 (incomplete 12 h history),
  (b) t >= t_onset (not at risk), (c) t + 12 > stay_end (insufficient
  follow-up for the 12 h horizon).

No future information (including the label series itself beyond the
prediction time) enters the feature tensor X(t).
"""
from __future__ import annotations

import numpy as np

from src.features.config import HORIZONS, LOOKBACK


def onset_hour(labels: np.ndarray, iculos: np.ndarray) -> float:
    """First ICULOS with SepsisLabel == 1 (NaN if never septic)."""
    idx = np.flatnonzero(labels == 1)
    return float(iculos[idx[0]]) if len(idx) else float("nan")


def window_label_mask(
    n_hours: int,
    onset: float,
    stay_end: int,
    horizon: int = HORIZONS["y12"],
) -> tuple[np.ndarray, np.ndarray]:
    """Return (label, keep) arrays over candidate hours 0..n_hours-1.

    label : 1 if onset in (t, t+H]  (0 otherwise)
    keep  : window is valid AND evaluable
            (complete 12 h history, at-risk, complete future horizon)
    """
    t = np.arange(n_hours)
    label = np.zeros(n_hours, dtype=np.int8)
    keep = np.zeros(n_hours, dtype=bool)
    complete_history = t >= (LOOKBACK - 1)
    complete_future = t <= (stay_end - horizon)
    at_risk = np.isnan(onset) if np.isnan(onset) else t < onset
    keep = complete_history & at_risk & complete_future
    if not np.isnan(onset):
        label = ((t < onset) & (onset <= t + horizon)).astype(np.int8)
    return label, keep