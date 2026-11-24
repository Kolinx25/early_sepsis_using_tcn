"""Component B — Contrastive trajectory learning.

TS2Vec-inspired instance-level contrastive pretraining of the trajectory
encoder (Yue et al., AAAI 2022), adapted to short ICU windows:

* Two stochastic "views" of each 12 h window: (a) a random contiguous crop
  of >= 6 hours, (b) independent channel dropout on the clinical values.
  Cropping mirrors TS2Vec's subseries augmentation; channel dropout
  simulates the sporadic lab measurement reality of ICU data.
* InfoNCE loss with temperature tau: crops of the *same* window are
  positives, other windows in the batch are negatives.
* Pretraining is performed on the TRAIN split only, with class-blind
  sampling, then the encoder is fine-tuned under supervision.

Contrastive learning was chosen over, e.g., masked reconstruction, because
representations of *whole trajectories* (not per-timestep reconstruction)
drive the pooled classification heads, and TS2Vec demonstrates strong
generic time-series representation quality with this exact instance-
contrastive formulation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

MIN_CROP = 6


def augment_batch(x: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor,
                  generator: torch.Generator, p_channel: float = 0.2):
    """Random contiguous crop + channel dropout -> one augmented view.

    Crops are zero-padded back to the full window length T so that every
    view keeps a fixed input shape (padded tail = 'not measured')."""
    B, T, C = x.shape
    device = x.device
    crop_len = int(torch.randint(MIN_CROP, T + 1, (1,), generator=generator))
    if crop_len < T:
        start = int(torch.randint(0, T - crop_len + 1, (1,), generator=generator))
        xs, ms, ds = x[:, start:start + crop_len], mask[:, start:start + crop_len], \
            delta[:, start:start + crop_len]
        pad = T - crop_len
        xs = torch.cat([xs, torch.zeros(B, pad, C, device=device)], dim=1)
        ms = torch.cat([ms, torch.zeros(B, pad, mask.shape[2], device=device)], dim=1)
        ds = torch.cat([ds, torch.zeros(B, pad, delta.shape[2], device=device)], dim=1)
    else:
        xs, ms, ds = x, mask, delta
    keep = (torch.rand(C, generator=generator) > p_channel).to(device)
    # channel dropout applies to clinical values only (channels [0:34) of x);
    # masks/deltas stay informative about the dropped pattern
    xs = xs.clone()
    xs[:, :, :34] = xs[:, :, :34] * keep[:34].view(1, 1, -1)
    return xs, ms, ds


def info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Symmetric InfoNCE over L2-normalised embeddings."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) +
                   F.cross_entropy(logits.T, labels))


def pretrain_encoder(model, loader, epochs: int, lr: float, device: str,
                     temperature: float = 0.2, log_prefix: str = "cl"):
    """Contrastive pretraining of model.embed on the train loader.

    The fused embedding (post-gate for the full architecture) is the
    contrastive representation so the gate is also shaped by the objective.
    """
    import numpy as np
    gen = torch.Generator().manual_seed(1234)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        tot, n = 0.0, 0
        for x, mask, delta, _y6, _y12 in loader:
            x, mask, delta = x.to(device), mask.to(device), delta.to(device)
            xa, ma, da = augment_batch(x, mask, delta, gen)
            xb, mb, db = augment_batch(x, mask, delta, gen)
            # crop may change lengths between views; pad both to the max
            T = max(xa.shape[1], xb.shape[1])
            def pad(t, T):
                return F.pad(t, (0, 0, 0, T - t.shape[1]))
            z1 = model.embed(pad(xa, T), pad(ma, T), pad(da, T))
            z2 = model.embed(pad(xb, T), pad(mb, T), pad(db, T))
            loss = info_nce(z1, z2, temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(x)
            n += len(x)
        print(f"[{log_prefix}] epoch {ep + 1}/{epochs}  loss={tot / n:.4f}")
    return model