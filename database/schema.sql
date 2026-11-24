-- SepsiGuard PostgreSQL schema (research prototype)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS patients (
    patient_id   BIGINT PRIMARY KEY,
    dataset      TEXT,
    is_septic    BOOLEAN DEFAULT FALSE,
    onset_hour   REAL,
    first_seen_at  TIMESTAMPTZ DEFAULT now(),
    last_seen_at   TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS observations (
    id           BIGSERIAL PRIMARY KEY,
    patient_id   BIGINT NOT NULL REFERENCES patients(patient_id),
    iculos       INT NOT NULL,
    recorded_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    channels     JSONB NOT NULL,          -- 34 clinical channels (null = missing)
    UNIQUE (patient_id, iculos)
);
CREATE INDEX IF NOT EXISTS obs_patient_idx ON observations (patient_id, iculos);

CREATE TABLE IF NOT EXISTS predictions (
    id                      BIGSERIAL PRIMARY KEY,
    patient_id              BIGINT NOT NULL REFERENCES patients(patient_id),
    iculos                  INT NOT NULL,
    prob_6h                 REAL,
    prob_12h                REAL,
    risk_band              TEXT,
    alert_6h                BOOLEAN DEFAULT FALSE,
    alert_12h               BOOLEAN DEFAULT FALSE,
    model_version           TEXT NOT NULL,
    input_window_start      INT,           -- iculos of the first hour in the window
    prediction_status       TEXT NOT NULL, -- 'ok' | 'pending_history' | 'error'
    predicted_at            TIMESTAMPTZ DEFAULT now(),
    UNIQUE (patient_id, iculos, model_version)
);
CREATE INDEX IF NOT EXISTS pred_patient_idx ON predictions (patient_id, iculos);

CREATE TABLE IF NOT EXISTS alerts (
    id            BIGSERIAL PRIMARY KEY,
    patient_id    BIGINT NOT NULL REFERENCES patients(patient_id),
    prediction_id BIGINT REFERENCES predictions(id),
    horizon       TEXT NOT NULL,             -- 'y6' | 'y12'
    kind          TEXT NOT NULL,             -- 'new_alert' | 'risk_increase'
    probability   REAL,
    message        TEXT,
    acknowledged  BOOLEAN DEFAULT FALSE,
    created_at     TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS alerts_ack_idx ON alerts (acknowledged, created_at DESC);

CREATE TABLE IF NOT EXISTS replay_sessions (
    id          BIGSERIAL PRIMARY KEY,
    started_at  TIMESTAMPTZ DEFAULT now(),
    speed       REAL,                         -- simulated hours per real second
    status      TEXT DEFAULT 'running',
    n_patients  INT
);

CREATE TABLE IF NOT EXISTS system_events (
    id          BIGSERIAL PRIMARY KEY,
    level       TEXT DEFAULT 'info',
    message     TEXT,
    created_at  TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS events_ts_idx ON system_events (created_at DESC);