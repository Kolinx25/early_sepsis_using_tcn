# SepsiGuard — Progress Log

> Living document. Updated at the end of every phase. Enables resuming without repeating work.

| Phase | Status | Notes |
|---|---|---|
| 0. Audit + scaffold | ✅ Complete | See `results/reports/00_initial_audit.md` |
| 1. Data science & EDA | ✅ Complete | `src/data/audit.py`; figures in `results/figures/exploratory_analysis`, `missingness`; tables in `results/tables/*` |
| 2. Preprocessing & features | ✅ Complete | `src/data/preprocess.py`, `src/features/windows.py`; artifacts in `artifacts/preprocessing`, `artifacts/scalers` |
| 3. Prediction labels | ✅ Complete | `src/features/labels.py`; policy in `docs/data_dictionary.md` |
| 4. Baselines | ✅ Complete | `src/training/baselines.py`; results `results/tables/baseline_comparison` |
| 5. Proposed architecture + ablations | ✅ Complete | `src/models/tcn.py`, `src/models/missingness_gate.py`, `src/training/contrastive.py`, `src/training/train.py`; ablations table in `results/tables/model_comparison` |
| 6. Evaluation | ✅ Complete | `src/evaluation/evaluate.py`; figures/tables under `results/figures/evaluation`, `results/tables/final_evaluation`; metrics JSON in `results/metrics` |
| 7. Artifacts | ✅ Complete | `artifacts/model_registry/sepsiguard_v1.0/` |
| 8. Backend & inference | ✅ Complete | `app/backend/`; FastAPI + PostgreSQL + WebSocket |
| 9. Historical ICU replay | ✅ Complete | `app/backend/replay.py`; shared inference path |
| 10. Website | ✅ Complete | `app/frontend/`; React + TS + Tailwind + Recharts, 5 screens |
| 11. Testing | ✅ Complete | `tests/`; end-to-end acceptance test passing |
| 12. Deployment | ✅ Complete | `docker-compose.yml`, `app/deployment/`; docs complete |

## Phase summaries and acceptance evidence

### Phase 0 — Audit (2026-11-24)
- Verified: repo empty; prior project `Kolinx25/Early_Sepsis_Prediction` audited (reusable: feature config, TCN block, split ratios; flawed: at-risk population, censoring, thresholding, calibration).
- Dataset obtained from public mirror, schema-verified: 40,336 patients (20,336 A + 20,000 B), 315 MB at `data/raw/physionet_2019/`.
- Environment: Python 3.14, torch 2.15 CPU, 4 cores, 15 GB RAM. Documented compute constraints → compact TCN + negative subsampling (training only).

### Phase 1 — EDA (2026-11-24)
- `dataset_summary` tables: **40,336 patients, 1,552,210 hourly observations, 1,825,756 raw CSV rows? (see tables)**; patient-level sepsis prevalence **7.51% (3,028 septic patients)**; hourly SepsisLabel prevalence 1.79%; set A vs set B differ in hospital mix and Unit1/Unit2 encoding, not in vital distributions (documented + tests).
- Missingness quantified per channel (median >60% for most labs) and shown to be **informative**: septic-vs-non-septic patients differ significantly in observation rates (figures + χ²/MW tests in `results/tables/descriptive_statistics`).
- Acceptance: ✔ all figures/tables saved, no leakage risk at this stage.

### Phase 2 — Preprocessing (2026-11-24)
- Reproducible `preprocess.py`: chronological sort, duplicate drop, missingness masks, forward-fill + population-median imputation, time-since-last-observation (delta) channels, z-normalisation fitted on **train patients only**.
- Artifacts: `artifacts/preprocessing/preprocess_config.json`, `artifacts/scalers/normalizer.json`.

### Phase 3 — Labels (2026-11-24)
- Operational onset: first hour of any run of `SepsisLabel == 1` within a patient stay (PhysioNet convention).
- Y6/Y12 = "onset occurs within (t, t+H]", computed from future only; **at-risk population = hours strictly before onset** (post-onset hours excluded); **censoring policy: windows with incomplete future horizon are excluded, never labelled negative** (documented).

### Phase 4 — Baselines (2026-11-24)
- Logistic regression, random forest, XGBoost on flattened 12×69 windows (train subsample documented), evaluated on full val/test.

### Phases 5–6 — Model + evaluation (2026-11-24)
- Full model: TCN encoder + TS2Vec-style contrastive pretraining + informative-missingness gating branch (mask+delta) + gated fusion + dual heads (Y6, Y12); Platt-calibrated on validation.
- Ablation table + comparison table + curves saved; bootstrap CIs; final test metrics in `results/metrics/final_evaluation.json`.

### Phases 7–12 — System (2026-11-24)
- Model registry v1.0 with inference manifest; FastAPI service with per-observation ingestion, streaming inference, WebSocket broadcast; replay engine; React dashboard (5 screens); pytest suite incl. end-to-end acceptance; docker-compose.