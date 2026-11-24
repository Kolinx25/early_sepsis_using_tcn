"""Phase 4 — Tabular baselines (Logistic Regression, Random Forest,
XGBoost) on flattened 12x69 prediction windows.

Protocol: trained on the same TRAIN subsample as the TCN variants (all
at-risk positives + 3x negatives), evaluated on the FULL validation/test
splits. One model per horizon (y6, y12).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.training.data import WindowData

ROOT = Path(__file__).resolve().parents[2]
WIN = ROOT / "data" / "processed" / "temporal_windows"
OUT = ROOT / "results" / "tables" / "baseline_comparison"
PRED = ROOT / "results" / "metrics" / "predictions"


def flatten(data) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (data.X.reshape(len(data.X), -1).astype(np.float32),
            data.y6.astype(int), data.y12.astype(int))


def run_baselines(save: bool = True):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    train_full = WindowData(str(WIN / "train.npz"))
    val = WindowData(str(WIN / "validation.npz"))
    test = WindowData(str(WIN / "test.npz"))
    sub = train_full.subsample(3.0, 42)
    Xtr, y6tr, y12tr = flatten(sub)
    Xva, y6va, y12va = flatten(val)
    Xte, y6te, y12te = flatten(test)
    print(f"baseline train {Xtr.shape}, val {Xva.shape}, test {Xte.shape}")
    PRED.mkdir(parents=True, exist_ok=True)

    models = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=3000, class_weight="balanced", C=1.0, n_jobs=-1),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, max_depth=12, n_jobs=-1, class_weight="balanced_subsample",
            random_state=42),
        "xgboost": lambda: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, tree_method="hist",
            scale_pos_weight=3.0, n_jobs=4, random_state=42, eval_metric="aucpr"),
    }
    results = {}
    for name, mk in models.items():
        results[name] = {}
        for horizon, (ytr, yva, yte) in {
                "y6": (y6tr, y6va, y6te), "y12": (y12tr, y12va, y12te)}.items():
            print(f"training {name} [{horizon}] ...")
            m = mk()
            m.fit(Xtr, ytr)
            pva = m.predict_proba(Xva)[:, 1]
            pte = m.predict_proba(Xte)[:, 1]
            results[name][horizon] = {
                "val": {"auroc": float(roc_auc_score(yva, pva)),
                         "auprc": float(average_precision_score(yva, pva)),
                         "prevalence": float(yva.mean())},
                "test": {"auroc": float(roc_auc_score(yte, pte)),
                         "auprc": float(average_precision_score(yte, pte)),
                         "prevalence": float(yte.mean())},
            }
            r = results[name][horizon]
            print(f"  {name}/{horizon}: val AUROC={r['val']['auroc']:.4f} "
                  f"test AUROC={r['test']['auroc']:.4f} test AUPRC={r['test']['auprc']:.4f}")
            np.save(PRED / f"{name}_test_{horizon}.npy", pte.astype(np.float32))
    np.save(PRED / f"labels_test_y6.npy", y6te)
    np.save(PRED / f"labels_test_y12.npy", y12te)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        PRED.mkdir(parents=True, exist_ok=True)
        (OUT / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_baselines()