"""Component A — Temporal Convolutional Network encoder.

Causal dilated temporal convolution blocks following Bai, Kolter & Koltun
(2018; arXiv:1803.01271 — preprint, disclosed as such in references.bib).
Causality is enforced by removing the receptive-field padding from the
*future* side after each convolution, so position t depends only on
inputs <= t (no future leakage).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dilation: int = 1, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad,
                              dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self._cut = pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self._cut:
            x = x[:, :, : -self._cut]  # strictly causal
        x = self.norm(x)
        return self.dropout(self.relu(x))


class TCNEncoder(nn.Module):
    """(B, T, C) -> (B, repr_dim) trajectory embedding.

    Mean+max temporal pooling yields a permutation-robust summary of the
    causal feature maps.
    """

    def __init__(self, in_channels: int = 69, channels: tuple[int, ...] = (64, 64, 64),
                 kernel_size: int = 3, dropout: float = 0.2):
        super().__init__()
        dilations = [1, 2, 4, 8, 16][: len(channels)]
        layers = []
        c_in = in_channels
        for c_out, d in zip(channels, dilations):
            layers.append(TemporalBlock(c_in, c_out, kernel_size, d, dropout))
            c_in = c_out
        self.net = nn.Sequential(*layers)
        self.out_dim = channels[-1] * 2  # mean ‖ max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x.transpose(1, 2))          # (B, C, T)
        return torch.cat([h.mean(dim=2), h.max(dim=2).values], dim=1)  # (B, 2C)