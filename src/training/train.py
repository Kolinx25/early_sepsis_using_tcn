"""Phases 4–5 — Train all model variants.

Variants
    tcn        : TCN encoder + dual heads (no gate, no contrastive)
    tcn_cl     : TCN + contrastive pretraining
    tcn_gate   : TCN + informative missingness gating (no contrastive)
    full       : TCN + contrastive pretraining + missingness gating (proposed)

Training protocol (identical across variants)
    * TRAIN split: all at-risk positive windows + 3x negatives (imbalance
      control; documented). VALIDATION/TEST: full splits, untouched.
    * Loss: weighted BCE per head (pos weight from the training subsample).
    * Adam, lr 1e-3, batch 512, max 12 epochs, early stop on validation
      mean AUPRC(y6, y12), seed fixed.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.sepsiguard import SepsiGuardNet
from src.training.contrastive import pretrain_encoder
from src.training.data import IndexedSubset, TorchWindows, WindowData, predict_logits

ROOT = Path(__file__).resolve().parents[2]
WIN = ROOT / "data" / "processed" / "temporal_windows"
CKPT = ROOT / "artifacts" / "checkpoints"

DEVICE = "cpu"
SEED = 42
NEG_POS_RATIO = 3.0
BATCH = 512
EPOCHS = 15
LR = 5e-4
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
PATIENCE = 5


def pos_weights(y6: np.ndarray, y12: np.ndarray):
    n = len(y6)
    return float((y6 == 0).sum() / max((y6 == 1).sum(), 1)), \
           float((y12 == 0).sum() / max((y12 == 1).sum(), 1))


def evaluate_val(model, val: WindowData) -> float:
    from scipy.special import expit
    from sklearn.metrics import average_precision_score
    l6, l12 = predict_logits(model, val, DEVICE)
    s6 = average_precision_score(val.y6, expit(l6.astype(np.float64)))
    s12 = average_precision_score(val.y12, expit(l12.astype(np.float64)))
    return 0.5 * (s6 + s12)


def train_variant(name: str, use_gate: bool, contrastive: bool,
                  save: bool = True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_full = WindowData(str(WIN / "train.npz"))
    val = WindowData(str(WIN / "validation.npz"))
    sub = train_full.subsample(NEG_POS_RATIO, SEED)
    print(f"[{name}] train subsample: {len(sub):,} windows "
          f"(y12+={sub.y12.sum():,}, y6+={sub.y6.sum():,})")

    w6, w12 = pos_weights(sub.y6, sub.y12)
    model = SepsiGuardNet(use_gate=use_gate, dropout=DROPOUT)

    if contrastive:
        loader = DataLoader(TorchWindows(sub), batch_size=BATCH, shuffle=True)
        pretrain_encoder(model, loader, epochs=8, lr=LR, device=DEVICE,
                         log_prefix=name)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    bce6 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w6))
    bce12 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w12))
    gloader = DataLoader(TorchWindows(sub), batch_size=BATCH, shuffle=True)

    best, best_state, bad = -1.0, None, 0
    for ep in range(EPOCHS):
        model.train()
        for x, mask, delta, y6, y12 in gloader:
            x, mask, delta = x.to(DEVICE), mask.to(DEVICE), delta.to(DEVICE)
            o6, o12 = model(x, mask, delta)
            loss = bce6(o6, y6.float().to(DEVICE)) + bce12(o12, y12.float().to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        score = evaluate_val(model, val)
        print(f"[{name}] epoch {ep + 1}: val mean AUPRC={score:.4f}")
        if score > best:
            best, bad = score, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[{name}] early stop")
                break
    model.load_state_dict(best_state)

    if save:
        CKPT.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "variant": name,
                    "use_gate": use_gate, "contrastive": contrastive,
                    "dropout": DROPOUT, "epochs_run": ep + 1, "val_mean_auprc": best,
                    "pos_weight": [w6, w12], "seed": SEED},
                   CKPT / f"{name}.pt")
    return model


VARIANTS = {
    "tcn": dict(use_gate=False, contrastive=False),
    "tcn_cl": dict(use_gate=False, contrastive=True),
    "tcn_gate": dict(use_gate=True, contrastive=False),
    "full": dict(use_gate=True, contrastive=True),
}

if __name__ == "__main__":
    import sys
    names = sys.argv[1:] or list(VARIANTS)
    for n in names:
        train_variant(n, **VARIANTS[n])