"""SepsiGuard network: TCN encoder + informative missingness gating + dual
prediction heads (6 h / 12 h horizons). Ablation variants share this module
via the `use_gate` flag and the contrastive pretraining in
src/training/contrastive.py.

    variant = "tcn"          : encoder + heads                      (baseline)
    variant = "tcn_cl"       : same, contrastively pretrained         (ablation)
    variant = "tcn_gate"     : encoder + gate + heads                (ablation)
    variant = "full"        : gate + contrastive pretraining        (proposed)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.missingness_gate import GatedFusion, InformativeMissingnessGate
from src.models.tcn import TCNEncoder

N_CLINICAL = 34          # channels [0:34)  values
MASK_SLICE = slice(34, 68)  # channels [34:68) missingness indicators


class SepsiGuardNet(nn.Module):
    def __init__(self, in_channels: int = 69, n_gate_channels: int = 34,
                 horizon_len: int = 12, tcn_channels: tuple[int, ...] = (64, 64, 64),
                 use_gate: bool = True, dropout: float = 0.2):
        super().__init__()
        self.encoder = TCNEncoder(in_channels, tcn_channels, dropout=dropout)
        repr_dim = self.encoder.out_dim
        self.use_gate = use_gate
        if use_gate:
            self.gate = InformativeMissingnessGate(n_gate_channels, horizon_len,
                                                    hidden=128, repr_dim=repr_dim,
                                                    dropout=dropout)
            self.fusion = GatedFusion(repr_dim)
        self.head_y6 = nn.Linear(repr_dim, 1)
        self.head_y12 = nn.Linear(repr_dim, 1)

    def embed(self, x: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 69), mask/delta: (B, T, 34) -> fused representation."""
        r_t = self.encoder(x)
        if not self.use_gate:
            return r_t
        r_m, g = self.gate(mask, delta)
        return self.fusion(r_t, r_m, g)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor):
        h = self.embed(x, mask, delta)
        return self.head_y6(h).squeeze(-1), self.head_y12(h).squeeze(-1)

    @staticmethod
    def split_input(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(B, T, 69) -> values+mask+iculos, mask, delta(None placeholder).

        Kept for API symmetry; delta is passed explicitly by the dataset."""
        mask = x[:, :, MASK_SLICE]
        return x, mask, None