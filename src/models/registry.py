"""Phase 7 — Package the deployed model into a versioned registry entry.

Registry layout:
    artifacts/model_registry/<version>/
        model.pt          deployed weights (+ variant flags)
        manifest.json     complete inference manifest:
                          feature order, normalisation, imputation policy,
                          delta cap, horizons, Platt calibration, thresholds,
                          training metadata, evaluation summary.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "artifacts" / "checkpoints"
REG = ROOT / "artifacts" / "model_registry"
VERSION = "sepsiguard_v1.0"


def package(deployed_variant: str = "tcn_gate") -> Path:
    out = REG / VERSION
    out.mkdir(parents=True, exist_ok=True)
    preprocess = json.loads((ROOT / "artifacts" / "preprocessing"
                             / "preprocess_config.json").read_text())
    final_eval = json.loads((ROOT / "results" / "metrics"
                             / "final_evaluation.json").read_text())
    ck = json.loads(json.dumps({}))  # placeholder
    import torch
    ck = torch.load(CKPT / f"{deployed_variant}.pt", map_location="cpu",
                    weights_only=False)
    shutil.copy(CKPT / f"{deployed_variant}.pt", out / "model.pt")

    manifest = {
        "model_version": VERSION,
        "research_title": ("Dynamic Early Sepsis Prediction via Contrastive "
                           "Trajectory Embeddings and Informative Missingness Gating"),
        "deployed_variant": deployed_variant,
        "variant_description": {
            "tcn": "TCN encoder + dual heads",
            "tcn_cl": "TCN + contrastive trajectory pretraining",
            "tcn_gate": "TCN + informative missingness gating",
            "full": "TCN + contrastive pretraining + missingness gating (complete architecture)",
        }[deployed_variant],
        "model_config": {
            "tcn_channels": [64, 64, 64], "kernel_size": 3,
            "dilations": [1, 2, 4], "dropout": ck.get("dropout", 0.2),
            "causal": True, "temporal_pool": "mean+max concat",
            "use_gate": ck["use_gate"], "contrastive_pretrained": ck["contrastive"],
        },
        "feature_order": preprocess["feature_order"],
        "input_spec": {
            "n_channels": 69,
            "lookback_hours": preprocess["lookback_hours"],
            "values": "34 clinical channels (imputed, z-normalised)",
            "missingness": "34 binary indicators (1 = absent)",
            "temporal": "ICULOS z-normalised",
            "gate_branch": "mask (12x34) + delta (12x34) hours-since-observation",
        },
        "horizons": preprocess["horizons"],
        "imputation": preprocess["imputation"],
        "normalisation": preprocess["normalisation"],
        "normalizer": preprocess["normalizer"],
        "delta_cap_hours": preprocess["delta_cap_hours"],
        "censoring_policy": preprocess["censoring_policy"],
        "at_risk_policy": preprocess["at_risk_policy"],
        "training": {
            "seed": ck["seed"], "epochs_run": ck["epochs_run"],
            "pos_weight_y6_y12": ck["pos_weight"],
            "train_subsample": "all positives + 3x negatives (train split only)",
            "early_stopping": "validation mean AUPRC, patience 5",
        },
        "platt": {h: final_eval[f"test_{h}"]["platt"] for h in ("y6", "y12")},
        "thresholds": {
            h: {"youden": final_eval[f"test_{h}"]["threshold_youden"],
                "alert": final_eval[f"test_{h}"]["alert_threshold_sensitivity80"]}
            for h in ("y6", "y12")
        },
        "evaluation_summary": {
            h: {k: final_eval[f"test_{h}"][k]
                for k in ("auroc", "auprc", "sensitivity", "specificity", "brier",
                          "auroc_ci95_patient_bootstrap",
                          "auprc_ci95_patient_bootstrap")}
            for h in ("y6", "y12")
        },
        "provenance": {
            "dataset": "PhysioNet/CinC Challenge 2019 (training sets A and B)",
            "at_risk_window_protocol": "pre-onset hours, complete 12h future horizon",
            "test_windows": 108207, "test_patients": 4600,
        },
        "intended_use": ("Research prototype for clinical decision support "
                          "demonstration. NOT a diagnostic or autonomous "
                          "treatment system."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"packaged {out}")
    return out


if __name__ == "__main__":
    package()