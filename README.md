# SepsiGuard

Early sepsis prediction for ICU patients using temporal deep learning.

SepsiGuard reads an hourly stream of vital signs and lab results, then estimates how likely a patient is to develop sepsis in the next 6 or 12 hours. It is built around a compact Temporal Convolutional Network (TCN) with two extras:

- **Contrastive trajectory learning** — the model learns what a "normal" versus "deteriorating" patient trajectory looks like, even when labels are scarce.
- **Informative missingness gating** — most ICU measurements are missing most of the time. Instead of pretending the gaps do not exist, the model uses a separate branch that pays attention to *which* values are missing and *how long* it has been since they were last seen.

This repository contains the full research pipeline, from raw data to a working web application.

---

## What problem this solves

Sepsis is a life-threatening response to infection. Catching it early matters, but ICU data is messy: measurements arrive at irregular times, many values are missing, and the signal is subtle. SepsiGuard turns this into a supervised early-warning problem and provides calibrated probability scores that clinicians (or downstream alert systems) can act on.

---

## Dataset

This project uses the **PhysioNet/Computing in Cardiology Challenge 2019** dataset, which contains hourly ICU observations for 40,336 patients.

You can obtain the exact dataset from the official source:

- **Official:** https://physionet.org/content/challenge-2019/1.0.0/  
  Requires a free PhysioNet account and agreement to the data-use agreement.

During development, the data was also verified against this public mirror of the challenge files:

- **Mirror used for verification:** https://github.com/MartinOravecSvK/Early-Prediction-of-Sepsis/tree/main/Dataset

Once you have the files, place them under:

```text
data/raw/physionet_2019/training_setA/
data/raw/physionet_2019/training_setB/
```

Expected contents: 20,336 patient files in `training_setA` and 20,000 patient files in `training_setB`, one CSV per patient, with 40 columns including vitals, labs, demographics, and a `SepsisLabel` flag.

---

## Quick start

The fastest way to run the full system is with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

Then open:

- Web dashboard: http://localhost:3000
- API docs: http://localhost:8000/docs
- WebSocket stream: ws://localhost:8000/ws/stream

To run the test suite locally (without Docker):

```bash
pip install -r requirements.txt
pytest
```

---

## Repository layout

```text
.
├── app/                    # Full-stack application
│   ├── backend/            # FastAPI service, PostgreSQL, WebSocket, replay engine
│   ├── frontend/           # React + TypeScript dashboard
│   └── deployment/         # Dockerfiles
├── src/                    # Research pipeline
│   ├── data/               # Loading and exploratory analysis
│   ├── features/           # Windowing, labels, missingness features
│   ├── models/             # TCN, gating branch, inference code
│   ├── training/           # Baselines, contrastive pretraining, full training
│   └── evaluation/         # Metrics, calibration, bootstrap CIs
├── tests/                  # pytest suite including end-to-end acceptance test
├── artifacts/              # Small, versioned artifacts (scaler, manifest)
├── database/               # PostgreSQL schema
├── results/                # Figures, tables, and final evaluation metrics
├── docker-compose.yml      # One-command deployment
├── requirements.txt        # Python dependencies
└── PROGRESS.md             # Detailed phase-by-phase build log
```

Note: large files such as raw data, trained model weights, and `node_modules` are excluded from Git by `.gitignore`.

---

## How the model works

1. **Preprocessing**  
   Each patient stay is sorted by time, duplicates are removed, and missing values are forward-filled up to a limit. Any remaining gaps are filled with the population median. The model also receives a binary "missingness" mask and a "time since last observed" value for each clinical channel.

2. **Windowing**  
   The model looks back 12 hours at a time. For every valid 12-hour window, two labels are created:
   - **Y6**: did sepsis onset happen within the next 6 hours?
   - **Y12**: did sepsis onset happen within the next 12 hours?

   Hours that occur *after* sepsis has already begun are excluded from training and evaluation, so the model only learns to predict future onset, not already-visible sepsis.

3. **Architecture**  
   A 3-layer causal TCN encodes the clinical time series. A parallel gating branch encodes missingness patterns. The two are fused and fed into separate heads for the 6-hour and 12-hour horizons. The deployed variant is calibrated with Platt scaling on the validation set.

4. **Evaluation**  
   Models are scored with AUROC, AUPRC, sensitivity, specificity, precision, F1, and Brier score. Thresholds are chosen on the validation set; the test set is used only for the final numbers. Confidence intervals come from patient-level bootstrapping.

---

## Results on the held-out test set

Final model: **TCN + informative missingness gating**, calibrated.

| Horizon | AUROC | AUPRC | Sensitivity | Specificity | Brier |
|---|---|---|---|---|---|
| 6 hours  | 0.790 | 0.039 | 0.631 | 0.830 | 0.009 |
| 12 hours | 0.802 | 0.099 | 0.675 | 0.806 | 0.022 |

A full comparison against baselines (logistic regression, random forest, XGBoost) and ablations (TCN without gating, TCN without contrastive learning) is in `results/tables/`.

---

## Reproducing the research pipeline

1. Download the dataset from the official PhysioNet link above and place it in `data/raw/physionet_2019/`.
2. Run exploratory analysis: `python -m src.data.audit`
3. Preprocess: `python -m src.data.preprocess`
4. Build windows and labels: `python -m src.features.build`
5. Train baselines: `python -m src.training.baselines`
6. Train proposed model: `python -m src.training.train`
7. Evaluate: `python -m src.evaluation.evaluate`

For exact commands and hyperparameters, see the source files in `src/` and the progress log in `PROGRESS.md`.

---

## Tech stack

- **Modeling:** Python, PyTorch, NumPy, pandas, scikit-learn, XGBoost
- **Backend:** FastAPI, PostgreSQL, SQLAlchemy, WebSockets
- **Frontend:** React, TypeScript, Tailwind CSS, Recharts, Vite
- **Deployment:** Docker, Docker Compose
- **Testing:** pytest

---

## Citation

If you use this work, please cite the dataset:

> Reyna, M. A., Josef, C., Jeter, R., Shashikumar, S. P., Westover, M. B., Nemati, S., Clifford, G. D., & Sharma, A. (2020). Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 (version 1.0.0). PhysioNet. https://doi.org/10.13026/vc4e-7w65

---

## License

The code in this repository is provided for research and educational purposes. The dataset itself is governed by the PhysioNet data-use agreement.
