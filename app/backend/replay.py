"""Phase 9 — Historical ICU replay engine.

The PhysioNet 2019 dataset is retrospective; the replay engine simulates the
chronological arrival of real ICU observations of TEST-split patients
(never seen in training) and streams them through the SAME inference path
as the REST ingestion endpoint (service.ingest_observation). No
observations are fabricated: every streamed value is a genuine hourly record
from the dataset, replayed in order.

The stream is labelled as simulated live data throughout the API and UI.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import db
from .config import DEFAULT_REPLAY_SPEED, REPLAY_POOL
from .service import SERVICE


class ReplayEngine:
    def __init__(self) -> None:
        self.status = "idle"          # idle | running | paused | stopped
        self.speed = DEFAULT_REPLAY_SPEED   # simulated hours per real second
        self.session_id: int | None = None
        self.n_patients = 0
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------ data
    def load_pool(self) -> pd.DataFrame:
        df = pd.read_parquet(REPLAY_POOL)
        df = df.rename(columns={c: c for c in df.columns})
        return df

    # ---------------------------------------------------------------- control
    def start(self, speed: float | None = None, n_patients: int = 25) -> dict:
        """Prepare session (sync, DB bookkeeping); returns immediately.
        The asyncio task is scheduled by the async API layer."""
        if self.status == "running":
            return {"status": "running", "session_id": self.session_id}
        self.speed = float(speed or self.speed)
        self.n_patients = int(n_patients)
        db.execute("INSERT INTO replay_sessions (speed, n_patients) VALUES (%s,%s)",
                   (self.speed, self.n_patients))
        row = db.query("SELECT id FROM replay_sessions ORDER BY id DESC LIMIT 1")[0]
        self.session_id = int(row["id"])
        db.log_event("info", f"replay session {self.session_id} started "
                             f"(speed={self.speed} sim-h/s, patients={self.n_patients})")
        self.status = "running"
        self._task = None   # caller (async API layer) schedules the task
        return {"status": "running", "session_id": self.session_id,
                "speed": self.speed, "n_patients": self.n_patients}

    def pause(self) -> dict:
        if self.status == "running":
            self.status = "paused"
            db.log_event("info", f"replay session {self.session_id} paused")
        return {"status": self.status, "session_id": self.session_id}

    def resume(self) -> dict:
        if self.status == "paused":
            self.status = "running"
        return {"status": self.status, "session_id": self.session_id}

    def stop(self) -> dict:
        self.status = "stopped"
        if self._task:
            self._task.cancel()
        self._task = None
        db.execute("UPDATE replay_sessions SET status='stopped' WHERE id=%s",
                    (self.session_id,))
        db.log_event("info", f"replay session {self.session_id} stopped")
        out = {"status": "stopped", "session_id": self.session_id}
        self.session_id = None
        return out

    def set_speed(self, speed: float) -> dict:
        self.speed = float(speed)
        return {"status": self.status, "speed": self.speed}

    def info(self) -> dict:
        return {"status": self.status, "session_id": self.session_id,
                "speed": self.speed, "n_patients": self.n_patients}

    # ------------------------------------------------------------------- loop
    async def _run(self, session_id: int) -> None:
        import json
        pool = self.load_pool()
        rng = np.random.default_rng(123)
        pids = pool.patient_id.unique()
        # mix of septic and non-septic patients
        sep = pids[np.isin(pids, pool.loc[pool.SepsisLabel.eq(1), "patient_id"].unique())]
        non = np.setdiff1d(pids, sep)
        pick = np.concatenate([
            rng.choice(sep, min(max(self.n_patients // 3, 5), len(sep)), replace=False),
            rng.choice(non, min(self.n_patients - min(max(self.n_patients // 3, 5), len(sep)),
                                len(non)), replace=False),
        ])
        sub = pool[pool.patient_id.isin(pick)].sort_values("ICULOS")
        # global chronological stream across patients
        events = list(sub[["patient_id", "ICULOS"]].itertuples(index=False))
        last_hour = None
        n_sent = 0
        try:
            for pid, iculos in events:
                while self.status == "paused":
                    await asyncio.sleep(0.3)
                if self.status == "stopped":
                    break
                if last_hour is not None and self.speed > 0:
                    await asyncio.sleep(min((iculos - last_hour) / self.speed, 2.0))
                last_hour = iculos
                row = sub[(sub.patient_id == pid) & (sub.ICULOS == iculos)].iloc[0]
                channels = {
                    k: (None if pd.isna(row[k]) else float(row[k]))
                    for k in ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
                              "EtCO2", "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2",
                              "SaO2", "AST", "BUN", "Alkalinephos", "Calcium",
                              "Chloride", "Creatinine", "Bilirubin_direct",
                              "Glucose", "Lactate", "Magnesium", "Phosphate",
                              "Potassium", "Bilirubin_total", "TroponinI", "Hct",
                              "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets")
                }
                # run in a worker thread (model inference + DB are blocking);
                # errors must not kill the stream
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, SERVICE.ingest_observation, int(pid), int(iculos),
                        channels, "replay")
                except Exception as e:
                    db.log_event("error", f"replay ingest failed for patient {pid} "
                                           f"hour {iculos}: {e}")
                n_sent += 1
                if n_sent % 50 == 0:
                    db.execute("UPDATE replay_sessions SET status='running' WHERE id=%s",
                                (session_id,))
        except asyncio.CancelledError:
            pass
        self.status = "idle" if self.status != "stopped" else "stopped"
        db.log_event("info", f"replay stream finished ({n_sent} observations)")


ENGINE = ReplayEngine()