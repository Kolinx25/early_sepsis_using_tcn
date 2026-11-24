"""Ingestion service: raw observation -> persistence -> features -> model
inference -> prediction persistence -> alerting -> WebSocket broadcast.

This is the SINGLE inference path: historical replay (app/backend/replay.py)
and the REST ingestion endpoint both route through ingest_observation().
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import numpy as np
import pandas as pd

from . import db
from .config import MODEL_VERSION, REGISTRY_DIR
from src.features.config import CLINICAL_FEATURES, LOOKBACK
from src.models.inference import SepsiGuardPredictor

RAW_TO_FEATURE = {
    "HR": "hr", "O2Sat": "o2sat", "Temp": "temp", "SBP": "sbp", "MAP": "map",
    "DBP": "dbp", "Resp": "resp", "EtCO2": "etco2", "BaseExcess": "baseexcess",
    "HCO3": "hco3", "FiO2": "fio2", "pH": "ph", "PaCO2": "paco2", "SaO2": "sao2",
    "AST": "ast", "BUN": "bun", "Alkalinephos": "alkalinephos",
    "Calcium": "calcium", "Chloride": "chloride", "Creatinine": "creatinine",
    "Bilirubin_direct": "bilirubin_direct", "Glucose": "glucose",
    "Lactate": "lactate", "Magnesium": "magnesium", "Phosphate": "phosphate",
    "Potassium": "potassium", "Bilirubin_total": "bilirubin_total",
    "TroponinI": "troponini", "Hct": "hct", "Hgb": "hgb", "PTT": "ptt",
    "WBC": "wbc", "Fibrinogen": "fibrinogen", "Platelets": "platelets",
    "ICULOS": "iculos",
}


class Broadcaster:
    """Fan-out JSON events to connected WebSocket clients."""

    def __init__(self) -> None:
        self.clients: set = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def attach(self, ws) -> None:
        self.clients.add(ws)

    def detach(self, ws) -> None:
        self.clients.discard(ws)

    def broadcast(self, event: dict) -> None:
        if not self.clients or self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(event), self.loop)

    async def _broadcast(self, event: dict) -> None:
        import json
        data = json.dumps(event, default=str)
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.detach(ws)


class IngestionService:
    def __init__(self) -> None:
        self.predictor = SepsiGuardPredictor(MODEL_VERSION, REGISTRY_DIR)
        self.broadcaster = Broadcaster()
        # last alert state per (patient, horizon) to emit state-change alerts
        self._alert_state: dict[tuple[int, str], bool] = {}
        self._history: dict[int, pd.DataFrame] = {}

    # ------------------------------------------------------------- ingestion
    def _patient_history(self, pid: int) -> pd.DataFrame:
        if pid not in self._history:
            rows = db.query(
                "SELECT channels, iculos FROM observations WHERE patient_id=%s "
                "ORDER BY iculos", (pid,))
            if rows:
                df = pd.DataFrame([dict(r["channels"], iculos=r["iculos"]) for r in rows])
                for c in CLINICAL_FEATURES:
                    if c not in df.columns:
                        df[c] = np.nan
                self._history[pid] = df[["iculos"] + CLINICAL_FEATURES]
            else:
                self._history[pid] = pd.DataFrame(columns=["iculos"] + CLINICAL_FEATURES)
        return self._history[pid]

    def ingest_observation(self, pid: int, iculos: int, channels: dict[str, Any],
                           source: str = "api") -> dict | None:
        """Persist one raw hourly observation, then run streaming inference."""
        # 1. persist raw observation
        db.execute(
            """INSERT INTO patients (patient_id) VALUES (%s)
               ON CONFLICT (patient_id) DO UPDATE SET last_seen_at = now()""",
            (pid,))
        clean = {RAW_TO_FEATURE.get(k, k): (None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v))
                 for k, v in channels.items() if RAW_TO_FEATURE.get(k, k) in CLINICAL_FEATURES}
        db.execute(
            "INSERT INTO observations (patient_id, iculos, channels) VALUES (%s,%s,%s)"
            " ON CONFLICT (patient_id, iculos) DO NOTHING",
            (pid, int(iculos), db.as_json(clean)))
        # 2. update in-memory history
        hist = self._patient_history(pid)
        row = {c: clean.get(c) for c in CLINICAL_FEATURES} | {"iculos": int(iculos)}
        hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
        hist = hist.drop_duplicates(subset="iculos", keep="last").sort_values("iculos")
        self._history[pid] = hist

        # 3. inference
        result = self.predictor.predict_history(hist)
        status = "ok"
        if result is None:
            prediction_id = self._store_pending(pid, iculos)
            event = {"type": "observation", "patient_id": pid, "iculos": int(iculos),
                     "prediction_status": "pending_history", "source": source}
            self.broadcaster.broadcast(event)
            return {"prediction_status": "pending_history",
                    "prediction_id": prediction_id, **event}
        p6, p12 = result["prob_6h"], result["prob_12h"]
        band = self.predictor.risk_band(p12)

        # 4. persist prediction
        with db.cursor() as (cur, c):
            cur.execute(
                """INSERT INTO predictions
                   (patient_id, iculos, prob_6h, prob_12h, risk_band, alert_6h,
                    alert_12h, model_version, input_window_start, prediction_status)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (patient_id, iculos, model_version) DO UPDATE
                     SET prob_6h=EXCLUDED.prob_6h, prob_12h=EXCLUDED.prob_12h,
                         risk_band=EXCLUDED.risk_band, alert_6h=EXCLUDED.alert_6h,
                         alert_12h=EXCLUDED.alert_12h
                   RETURNING id""",
                (pid, int(iculos), p6, p12, band, result["alert_6h"],
                 result["alert_12h"], self.predictor.version,
                 int(iculos) - LOOKBACK + 1, status))
            prediction_id = cur.fetchone()["id"]
            c.commit()

        # 5. alert state changes -> alerts table + events
        new_alerts = []
        for horizon, p, alert in (("y6", p6, result["alert_6h"]),
                                  ("y12", p12, result["alert_12h"])):
            prev = self._alert_state.get((pid, horizon), False)
            if alert and not prev:
                kind = "new_alert"
                msg = (f"Patient {pid}: {int((6 if horizon=='y6' else 12))}h sepsis "
                       f"risk {p:.3f} above alert threshold")
                with db.cursor() as (cur, c):
                    cur.execute(
                        """INSERT INTO alerts (patient_id, prediction_id, horizon,
                           kind, probability, message) VALUES (%s,%s,%s,%s,%s,%s)
                           RETURNING id""",
                        (pid, prediction_id, horizon, kind, float(p), msg))
                    alert_id = cur.fetchone()["id"]
                    c.commit()
                new_alerts.append({"id": alert_id, "horizon": horizon,
                                   "kind": kind, "probability": float(p)})
            self._alert_state[(pid, horizon)] = alert

        event = {
            "type": "prediction", "source": source, "patient_id": pid,
            "iculos": int(iculos), "prob_6h": round(p6, 4),
            "prob_12h": round(p12, 4), "risk_band": band,
            "prediction_status": "ok",
            "alert_6h": bool(result["alert_6h"]), "alert_12h": bool(result["alert_12h"]),
            "model_version": self.predictor.version, "prediction_id": prediction_id,
            "new_alerts": new_alerts,
        }
        self.broadcaster.broadcast(event)
        return {"prediction_status": "ok", "prediction_id": prediction_id, **event}

    def _store_pending(self, pid: int, iculos: int) -> int | None:
        with db.cursor() as (cur, c):
            cur.execute(
                """INSERT INTO predictions (patient_id, iculos, model_version,
                   input_window_start, prediction_status)
                   VALUES (%s,%s,%s,%s,'pending_history')
                   ON CONFLICT (patient_id, iculos, model_version) DO NOTHING
                   RETURNING id""",
                (pid, int(iculos), self.predictor.version,
                 int(iculos) - LOOKBACK + 1))
            row = cur.fetchone()
            c.commit()
        return row["id"] if row else None


SERVICE = IngestionService()