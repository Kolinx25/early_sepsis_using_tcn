"""Component C/D — Informative missingness gating and feature fusion.

Motivation: Che et al. (2018, Sci Rep) showed that missingness patterns in
ICU multivariate time series are informative and exploitable (GRU-D).
Phase 1 of this project confirmed this empirically for PhysioNet 2019:
observation rates differ between eventual-sepsis and non-sepsis patients in
30/34 channels (Mann–Whitney U, Bonferroni-corrected), and missingness
indicators alone predict 12 h sepsis onset with AUROC 0.61.

The gate consumes, per prediction window:
    * the binary missingness mask (12 x 34)  — *what* was measured
    * the delta channels (12 x 34)          — *how long ago* it was measured
and produces a gating vector that interpolates between the temporal
trajectory representation r_t (Component A) and a mask-derived
representation r_m. Fusion:  h = σ(W_g r_m) ⊙ r_t + (1 - σ(W_g r_m)) ⊙ r_m.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class InformativeMissingnessGate(nn.Module):
    def __init__(self, n_channels: int = 34, horizon_len: int = 12,
                hidden: int = 128, repr_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        in_dim = n_channels * horizon_len * 2  # mask ‖ delta
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.r_m_proj = nn.Linear(hidden, repr_dim)
        self.gate_proj = nn.Linear(hidden, repr_dim)

    def forward(self, mask: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """mask, delta: (B, T, 34) -> (r_m (B, repr), gate (B, repr))."""
        flat = torch.cat([mask.flatten(1), delta.flatten(1)], dim=1)
        h = self.body(flat)
        return self.r_m_proj(h), torch.sigmoid(self.gate_proj(h))


class GatedFusion(nn.Module):
    def __init__(self, repr_dim: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(repr_dim)

    def forward(self, r_t: torch.Tensor, r_m: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return self.norm(gate * r_t + (1.0 - gate) * r_m)