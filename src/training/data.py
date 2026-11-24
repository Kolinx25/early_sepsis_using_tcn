"""Window datasets for TCN training/evaluation (streamed from .npz)."""
from __future__ import annotations

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

MASK_SLICE = slice(34, 68)


class WindowData:
    """In-memory split container."""

    def __init__(self, path: str):
        z = np.load(path)
        self.X, self.D = z["X"], z["D"]          # (N, 12, 69), (N, 12, 34)
        self.y6, self.y12 = z["y6"], z["y12"]
        self.pid, self.t = z["pid"], z["t"]

    def subsample(self, neg_pos_ratio: float, seed: int = 42):
        """Keep all 12 h-positive windows + neg_pos_ratio * n_pos random
        negatives. Applied to TRAINING only (documented imbalance control);
        evaluation always uses the full split."""
        rng = np.random.default_rng(seed)
        pos = np.flatnonzero((self.y12 == 1) | (self.y6 == 1))
        neg = np.flatnonzero((self.y12 == 0) & (self.y6 == 0))
        n_neg = min(len(neg), int(len(pos) * neg_pos_ratio))
        neg = rng.choice(neg, size=n_neg, replace=False)
        idx = np.sort(np.concatenate([pos, neg]))
        return IndexedSubset(self, idx)


class IndexedSubset:
    def __init__(self, data: WindowData, idx: np.ndarray):
        self.data, self.idx = data, idx
        for k in ("X", "D", "y6", "y12", "pid", "t"):
            setattr(self, k, getattr(data, k)[idx])

    def __len__(self):
        return len(self.idx)


class TorchWindows(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data.y12)

    def __getitem__(self, i):
        x = torch.from_numpy(self.data.X[i])               # (12, 69)
        mask = x[:, MASK_SLICE].clone()                    # (12, 34)
        delta = torch.from_numpy(self.data.D[i])           # (12, 34)
        return x, mask, delta, int(self.data.y6[i]), int(self.data.y12[i])


def predict_logits(model, data, device: str, batch: int = 1024):
    """Full-split inference -> (logits_y6, logits_y12)."""
    model.eval()
    ds = TorchWindows(data)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch, shuffle=False)
    l6, l12 = [], []
    with torch.no_grad():
        for x, mask, delta, _, _ in loader:
            o6, o12 = model(x.to(device), mask.to(device), delta.to(device))
            l6.append(o6.cpu())
            l12.append(o12.cpu())
    return torch.cat(l6).numpy(), torch.cat(l12).numpy()