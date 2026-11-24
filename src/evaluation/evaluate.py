"""Phase 6 — Model evaluation.

Protocol (per specification):
* Thresholds and Platt calibration are selected/fitted on the VALIDATION
  split; the TEST split is evaluated once, at the end, per model.
* Metrics: AUROC, AUPRC, sensitivity, specificity, precision, recall, F1,
  Brier, calibration (Platt), confusion matrix.
* Uncertainty: patient-level bootstrap 95% CIs for AUROC/AUPRC of the
  deployed model (resampling patients, respecting within-patient clustering).
* Early-warning lead time and clinical utility (net benefit / decision-curve
  style) are reported for the deployed model on test.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix,
    precision_recall_curve, roc_auc_score, roc_curve,
)

from src.models.sepsiguard import SepsiGuardNet
from src.training.data import WindowData, predict_logits

ROOT = Path(__file__).resolve().parents[2]
WIN = ROOT / "data" / "processed" / "temporal_windows"
CKPT = ROOT / "artifacts" / "checkpoints"
FIG = ROOT / "results" / "figures" / "evaluation"
TAB_CMP = ROOT / "results" / "tables" / "model_comparison"
TAB_FIN = ROOT / "results" / "tables" / "final_evaluation"
MET = ROOT / "results" / "metrics"
CAL = ROOT / "artifacts" / "calibration"
REG = ROOT / "artifacts" / "model_registry"
PRED = MET / "predictions"

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
VARIANTS = ["tcn", "tcn_cl", "tcn_gate", "full"]
VARIANT_LABEL = {"tcn": "TCN (baseline)", "tcn_cl": "TCN + contrastive",
                 "tcn_gate": "TCN + missingness gate", "full": "SepsiGuard (full)"}


def load_model(variant: str) -> SepsiGuardNet:
    ck = torch_load_compat(CKPT / f"{variant}.pt")
    model = SepisiGuardFromCheckpoint(ck)
    model.eval()
    return model


def torch_load_compat(path: Path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def SepisiGuardFromCheckpoint(ck) -> SepsiGuardNet:
    m = SepsiGuardNet(use_gate=ck["use_gate"], dropout=ck.get("dropout", 0.2))
    m.load_state_dict(ck["state_dict"])
    return m


def binary_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "precision": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "recall": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else float("nan"),
        "brier": float(brier_score_loss(y, p)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "prevalence": float(np.mean(y)),
    }


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def sensitivity_threshold(y, p, target=0.80):
    """Smallest threshold achieving >= target sensitivity (alerting rule)."""
    fpr, tpr, thr = roc_curve(y, p)
    ok = np.flatnonzero(tpr >= target)
    return float(thr[ok[0]]) if len(ok) else 0.0


def platt_fit(logits, y):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, max_iter=1000)
    lr.fit(logits.reshape(-1, 1), y)
    a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
    return lambda l: expit(a * l + b), {"a": a, "b": b}


def patient_bootstrap(y, p, pid, n_boot=200, seed=42):
    """Patient-level bootstrap CI for AUROC/AUPRC."""
    rng = np.random.default_rng(seed)
    up = np.unique(pid)
    aucs, aps = [], []
    for _ in range(n_boot):
        sel = rng.choice(up, size=len(up), replace=True)
        idx = np.concatenate([np.flatnonzero(pid == s) for s in sel])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        aps.append(average_precision_score(y[idx], p[idx]))
    return {"auroc_ci95": [float(x) for x in np.percentile(aucs, [2.5, 97.5])],
            "auprc_ci95": [float(x) for x in np.percentile(aps, [2.5, 97.5])]}


def net_benefit(y, p, thr):
    """Decision-curve net benefit: treat-if-above-threshold strategy."""
    pred = (p >= thr).astype(float)
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    n = float(len(y))
    return tp / n - fp / n * (thr / (1 - thr))


def main():
    import torch  # noqa
    val = WindowData(str(WIN / "validation.npz"))
    test = WindowData(str(WIN / "test.npz"))
    for d in (FIG, TAB_CMP, TAB_FIN, MET, CAL, REG, PRED):
        d.mkdir(parents=True, exist_ok=True)

    all_results = {}
    val_logits, test_logits = {}, {}
    for v in VARIANTS:
        model = load_model(v)
        l6v, l12v = predict_logits(model, val, "cpu")
        l6t, l12t = predict_logits(model, test, "cpu")
        val_logits[v] = {("y6",): l6v, ("y12",): l12v}
        test_logits[v] = {("y6",): l6t, ("y12",): l12t}
        all_results[v] = {}
        for h, (lv, lt, yv, yt, pidv, pidt) in {
            "y6": (l6v, l6t, val.y6, test.y6, val.pid, test.pid),
            "y12": (l12v, l12t, val.y6 * 0 + val.y12, test.y12, val.pid, test.pid),
        }.items():
            pv, pt_ = expit(lv.astype(np.float64)), expit(lt.astype(np.float64))
            # calibration + thresholds from VALIDATION only
            cal_fn, params = platt_fit(lv, yv)
            thr = youden_threshold(yv, cal_fn(lv))
            thr_alert = sensitivity_threshold(yv, cal_fn(lv), 0.80)
            all_results[v][h] = {
                "val": binary_metrics(yv, cal_fn(lv), thr),
                "test": binary_metrics(yt, cal_fn(lt), thr),
                "threshold_youden_calibrated": thr,
                "threshold_alert_sensitivity80": thr_alert,
                "platt": params,
            }
            np.save(PRED / f"{v}_test_{h}.npy", pt_.astype(np.float32))
            np.save(PRED / f"{v}_val_{h}.npy", pv.astype(np.float32))
        np.save(PRED / "labels_val_y6.npy", val.y6)
        np.save(PRED / "labels_val_y12.npy", val.y12)
        np.save(PRED / "labels_test_y6.npy", test.y6)
        np.save(PRED / "labels_test_y12.npy", test.y12)
        np.save(PRED / "pid_test.npy", test.pid)
        np.save(PRED / "t_test.npy", test.t)

    # -------- deployed variant = best validation mean AUPRC ------------------
    def val_mean(v):
        return 0.5 * (all_results[v]["y6"]["val"]["auprc"]
                       + all_results[v]["y12"]["val"]["auprc"])
    deployed = max(VARIANTS, key=val_mean)
    print("deployed variant:", deployed, "val mean AUPRC =", round(val_mean(deployed), 4))

    # -------- comparison table ------------------------------------------------
    rows = []
    for v in VARIANTS:
        for h in ("y6", "y12"):
            r = all_results[v][h]
            rows.append({"model": VARIANT_LABEL[v], "variant": v, "horizon": h,
                         "val_auroc": round(r["val"]["auroc"], 4),
                         "val_auprc": round(r["val"]["auprc"], 4),
                         "test_auroc": round(r["test"]["auroc"], 4),
                         "test_auprc": round(r["test"]["auprc"], 4),
                         "test_sensitivity": round(r["test"]["sensitivity"], 4),
                         "test_specificity": round(r["test"]["specificity"], 4),
                         "test_brier": round(r["test"]["brier"], 4)})
    # tabular baselines
    bl = json.loads((ROOT / "results" / "tables" / "baseline_comparison"
                     / "baseline_metrics.json").read_text())
    for name, hrs in bl.items():
        for h, m in hrs.items():
            rows.append({"model": name.replace("_", " "), "variant": name, "horizon": h,
                         "val_auroc": round(m["val"]["auroc"], 4),
                         "val_auprc": round(m["val"]["auprc"], 4),
                         "test_auroc": round(m["test"]["auroc"], 4),
                         "test_auprc": round(m["test"]["auprc"], 4),
                         "test_sensitivity": "", "test_specificity": "",
                         "test_brier": ""})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(TAB_CMP / "model_comparison.csv", index=False)
    abl = cmp_df[cmp_df.variant.isin(VARIANTS)]
    abl.to_csv(TAB_CMP / "ablation_table.csv", index=False)

    # -------- figures ---------------------------------------------------------
    for h, (yv, yt) in {"y6": (val.y6, test.y6), "y12": (val.y12, test.y12)}.items():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for v in VARIANTS:
            pt_ = np.load(PRED / f"{v}_test_{h}.npy")
            fpr, tpr, _ = roc_curve(yt, pt_)
            axes[0].plot(fpr, tpr, label=f"{VARIANT_LABEL[v]} (AUC={roc_auc_score(yt, pt_):.3f})")
        for name in ("logistic_regression", "random_forest", "xgboost"):
            p = np.load(PRED / f"{name}_test_{h}.npy")
            fpr, tpr, _ = roc_curve(yt, p)
            axes[0].plot(fpr, tpr, "--", alpha=0.6,
                         label=f"{name.replace('_',' ')} (AUC={roc_auc_score(yt, p):.3f})")
        axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
        axes[0].set(xlabel="false positive rate", ylabel="true positive rate",
                    title=f"ROC — test split, {h} horizon")
        axes[0].legend(fontsize=7, loc="lower right")
        for v in VARIANTS:
            pt_ = np.load(PRED / f"{v}_test_{h}.npy")
            pr, rc, _ = precision_recall_curve(yt, pt_)
            axes[1].plot(rc, pr, label=f"{VARIANT_LABEL[v]} "
                        f"(AP={average_precision_score(yt, pt_):.3f})")
        axes[1].axhline(yt.mean(), color="k", ls=":", lw=0.8,
                        label=f"prevalence ({yt.mean():.3f})")
        axes[1].set(xlabel="recall", ylabel="precision",
                    title=f"Precision–recall — test split, {h} horizon")
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"roc_pr_test_{h}.png")
        plt.close(fig)

    # calibration + confusion (deployed, both horizons)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    cms = {}
    for ax, h in zip(axes, ("y6", "y12")):
        r = all_results[deployed][h]
        lt = test_logits[deployed][(h,)]
        cal_fn = lambda l, a=r["platt"]["a"], b=r["platt"]["b"]: expit(a * l + b)
        p = cal_fn(lt.astype(np.float64))
        yt = test.y12 if h == "y12" else test.y6
        bins = np.quantile(p, np.linspace(0, 1, 11))
        bid = np.digitize(p, bins[1:-1])
        cx = [p[bid == i].mean() for i in range(10)]
        cy = [yt[bid == i].mean() for i in range(10)]
        ax.plot(cx, cy, "o-", label="deployed model")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set(xlabel="predicted probability (deciles)", ylabel="observed rate",
               title=f"Calibration — test, {h}")
        cm = confusion_matrix(yt, (p >= r["threshold_youden_calibrated"]).astype(int),
                              labels=[0, 1])
        cms[h] = cm.tolist()
        (TAB_FIN / f"confusion_matrix_{h}.csv").write_text(
            pd.DataFrame(cm, index=["actual 0", "actual 1"],
                         columns=["pred 0", "pred 1"]).to_csv())
    fig.tight_layout()
    fig.savefig(FIG / "calibration_test.png")
    plt.close(fig)

    # -------- bootstrap CIs (deployed) ---------------------------------------
    final = {"deployed_variant": deployed,
             "val_mean_auprc": round(val_mean(deployed), 4),
             "variants": {v: {"val_mean_auprc": round(val_mean(v), 4)} for v in VARIANTS}}
    for h in ("y6", "y12"):
        lt = test_logits[deployed][(h,)]
        r = all_results[deployed][h]
        p = expit(r["platt"]["a"] * lt + r["platt"]["b"]).astype(np.float64)
        yt = (test.y12 if h == "y12" else test.y6).astype(int)
        ci = patient_bootstrap(yt, p, test.pid)
        m = binary_metrics(yt, p, r["threshold_youden_calibrated"])
        nb = net_benefit(yt, p, r["threshold_youden_calibrated"])
        # lead time: for septic test patients, first alert before onset
        lead = []
        # onset per patient from raw label trajectory (recomputed for safety)
        obs = pd.read_parquet(ROOT / "data" / "interim" / "validated" / "observations.parquet",
                              columns=["patient_id", "ICULOS", "SepsisLabel"])
        onset_map = obs[obs.SepsisLabel == 1].groupby("patient_id")["ICULOS"].min()
        thr_alert = r["threshold_alert_sensitivity80"]
        for pid in np.unique(test.pid):
            if pid not in onset_map.index:
                continue
            onset = onset_map.loc[pid]
            mask = (test.pid == pid) & (test.t < onset)
            if not mask.any():
                continue
            pa = p[mask]
            ta = test.t[mask]
            hits = np.flatnonzero(pa >= thr_alert)
            if len(hits):
                lead.append(float(onset - ta[hits[0]]))
        final[f"test_{h}"] = {
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()},
            "auroc_ci95_patient_bootstrap": ci["auroc_ci95"],
            "auprc_ci95_patient_bootstrap": ci["auprc_ci95"],
            "net_benefit_at_youden_threshold": round(nb, 5),
            "platt": r["platt"],
            "threshold_youden": r["threshold_youden_calibrated"],
            "alert_threshold_sensitivity80": thr_alert,
            "median_lead_time_hours_when_alerting": (
                float(np.median(lead)) if lead else None),
            "n_septic_test_patients_with_alert": len(lead),
        }
    (MET / "final_evaluation.json").write_text(json.dumps(final, indent=2))
    (TAB_FIN / "final_test_metrics.json").write_text(
        json.dumps({"y6": final["test_y6"], "y12": final["test_y12"]}, indent=2))

    # -------- subgroup analysis (deployed, y12) --------------------------------
    p = expit(all_results[deployed]["y12"]["platt"]["a"] * test_logits[deployed][("y12",)]
              + all_results[deployed]["y12"]["platt"]["b"]).astype(np.float64)
    yt = test.y12.astype(int)
    pat = pd.read_parquet(ROOT / "data" / "interim" / "validated" / "patients.parquet")
    sub = pd.DataFrame({"pid": test.pid, "y": yt, "p": p})
    sub["age"] = sub.pid.map(dict(zip(pat.patient_id, pat.age)))
    sub["dataset"] = np.where(sub.pid > 50000, "setB", "setA")
    rows = []
    for name, mask in {
        "age<65": sub.age < 65, "age>=65": sub.age >= 65,
        "setA": sub.dataset == "setA", "setB": sub.dataset == "setB",
    }.items():
        m = mask.to_numpy()
        rows.append({"subgroup": name, "n_windows": int(m.sum()),
                     "auroc": float(roc_auc_score(sub.y[m], sub.p[m])),
                     "auprc": float(average_precision_score(sub.y[m], sub.p[m]))})
    pd.DataFrame(rows).to_csv(TAB_FIN / "subgroup_analysis.csv", index=False)

    print(json.dumps(final, indent=2)[:2000])
    return final


if __name__ == "__main__":
    main()