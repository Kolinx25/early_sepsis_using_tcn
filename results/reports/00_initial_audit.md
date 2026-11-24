# SepsiGuard — Initial Project Audit (Phase 0)

Date: 2026-11-24
Auditor: Automated development agent (acting as senior data scientist / biomedical ML researcher)

---

## 1. Environment inspected

- Target repository: `Kolinx25/early_sepsis_using_tcn` (this repo)
- Local PostgreSQL server: **none found** (only libpq client libraries installed)
- Working directory contents prior to this audit: a stub `README.md` and an empty git history (one initial commit). **No source code, notebooks, datasets, checkpoints, evaluation results or documentation existed in this repo.**

## 2. Prior project located and audited

A prior project by the same owner was found on GitHub: `Kolinx25/Early_Sepsis_Prediction` (cloned to `/tmp/prev_sepsis` for inspection). Contents:

| Item | Status | Notes |
|---|---|---|
| `scripts/python/feature_config.py` | **Relevant, correct** | Defines 34 clinical features + 34 missingness indicators + `iculos` = **69 model inputs**, lookback 12 h, horizons 6 h/12 h. Matches historical claims exactly. |
| `scripts/python/create_windows.py` | **Partially correct** | Builds 12 h windows and Y6/Y12 from future `SepsisLabel`; **flaw 1**: requires a *complete* 12 h future horizon → drops censored windows (no explicit censoring policy); **flaw 2**: does not exclude post-sepsis-onset hours from the at-risk population; **flaw 3**: no time-since-last-observation features; **flaw 4**: normalisation/calibration not evidenced. |
| `scripts/python/models/tcn.py` | **Reusable** | Causal dilated `TemporalBlock` with correct future-padding removal (`x[:, :, :-padding]`). Small 3-layer TCN (64-64-128), single logit head (Y12 only). No contrastive learning, no missingness gating, no Y6 head. |
| `scripts/python/models/train_tcn.py` | **Partially reusable** | BCE training loop, seed control, checkpointing. Threshold fixed at 0.5 (not validation-tuned); no calibration; no class-imbalance handling; single horizon only. |
| `scripts/sql/schema.sql`, `database.py`, `ingest.py` | **Partially reusable** | PostgreSQL warehouse design (patients, observations, trajectory windows). Will be adapted, not copied blindly. |
| `outputs/tables/table1_dataset_characteristics.csv` | **Historical reference** | 675,346 windows; 472,914 train / 101,977 val / 100,455 test; 21,462 / 4,599 / 4,600 patients (30,661 window-eligible); Y12 positive prevalence 2.45%. Matches the historical figures quoted in the project brief. |
| `outputs/tables/table3_performance_metrics.csv` | **Historical reference** | Prior TCN: AUROC 0.793, AUPRC 0.096, Brier 0.125 on test (Y12). Baseline to beat/verify. |
| `models/best_tcn_y12.pt` | **Not directly reusable** | 634 KB checkpoint trained under an unversioned preprocessing regime; will be retrained under the new, fully documented pipeline. |
| Figures (`eda.pdf`, ROC/PR/calibration/DCA PNGs) | **Reference only** | Will be regenerated reproducibly from scripts under version control. |

**Conclusion:** the prior project is a sound foundation but is incomplete relative to the SepsiGuard specification: no 6 h horizon model, no contrastive learning, no missingness gating, no fusion/ablations, no calibration, no backend/API, no replay engine, no website, no tests, no Docker deployment.

## 3. Dataset audit

- **Source:** PhysioNet/Computing in Cardiology Challenge 2019 (Reyna et al. 2020), `training_setA` (hospital A/B/C) and `training_setB` (hospital D/E).
- **Official access** is credentialed (PhysioNet account + DUA). No credentials are available in this environment, so the dataset was obtained from a public GitHub mirror of the challenge distribution (`MartinOravecSvK/Early-Prediction-of-Sepsis`, `Dataset/`) and **verified against the official challenge schema**: 40 columns (34 clinical channels, Age, Gender, Unit1, Unit2, HospAdmTime, ICULOS, SepsisLabel), hourly rows, `|`-separated.
- **Verified counts:** set A = 20,336 patient files; set B = 20,000 patient files; total = **40,336 patients** — consistent with the historical claim (~40,336 records).
- Files now stored at `data/raw/physionet_2019/training_set{A,B}/` (315 MB).

## 4. Compute environment audit (constrains experimental scale)

- 4 CPU cores, 15 GB RAM, ~19 GB free disk, **no GPU**. PyTorch 2.15 (CPU wheels) available on Python 3.14.
- Consequence (documented decision, not silent): deep models are trained with a **compact TCN** and **training-set negative subsampling** (imbalance control), a standard practice for rare-event ICU prediction; all evaluation (validation/test) is performed on **full, unmodified** window populations.

## 5. Audit verdict

| Question | Answer |
|---|---|
| What already exists? | A prior TCN pipeline (feature config, windowing, small TCN, SQL schema) in a separate repo; the raw dataset is available. |
| What works? | The 69-feature definition, causal TCN block, window construction, SQL schema concepts. |
| What is incomplete? | Everything downstream of a single-horizon TCN: Y6 modelling, contrastive learning, missingness gating, fusion, ablations, calibration, backend, replay, website, tests, deployment. |
| What is incorrect? | Post-onset hours treated as at-risk observations; censored windows silently dropped without policy; fixed 0.5 threshold; no leakage controls for normalisation; unversioned artifacts. |
| What needs rebuilding? | The full pipeline, in this repo, under the required structure, with reproducible scripts and documented decisions. |
| What can be reused? | Feature ordering (34+34+`iculos`), the causal TCN block design, the patient-level split ratios (70/15/15) that reproduce the historical window counts, the SQL schema skeleton. |

## 6. Fixed development plan (binding from here on)

0. ✅ Audit (this report) + scaffold + environment.
1. **Data science & EDA** — full-dataset audit, missingness, distributions, A-vs-B comparison; figures/tables saved.
2. **Preprocessing & features** — reproducible pipeline: chronological ordering, duplicate handling, missingness masks, time-since-observation, forward-fill/population imputation, **training-only** normalisation, 12 h windows; artifacts versioned.
3. **Labels** — Y6/Y12 with at-risk-population rules (post-onset exclusion), explicit censoring policy, documented operational onset definition (Sepsis-onset ≥ 1 consecutive SepsisLabel hour; window label = onset within horizon).
4. **Baselines** — Logistic Regression, Random Forest, XGBoost, plain TCN.
5. **Proposed architecture** — TCN + contrastive trajectory embeddings (TS2Vec-inspired InfoNCE pretraining) + informative missingness gating (mask/delta branch with multiplicative gating) + fused dual heads (6 h, 12 h). Ablations: −contrastive, −gating, −both.
6. **Evaluation** — AUROC, AUPRC, sensitivity, specificity, precision, recall, F1, Brier, calibration (per horizon), thresholds selected on validation, test untouched until final; bootstrap uncertainty.
7. **Artifacts** — versioned model registry with full inference manifest.
8. **Backend & inference** — FastAPI + PostgreSQL + WebSocket broadcast.
9. **Historical ICU replay** — chronological streaming engine feeding the same inference path.
10. **Website** — React + TypeScript + Tailwind + Recharts, 5 screens, live updates.
11. **Testing** — pytest suite + mandatory end-to-end acceptance test.
12. **Deployment** — Docker Compose (backend, frontend, PostgreSQL), health checks, docs.

Stopping rules: once the acceptance criteria of Phase 12 pass, development stops.