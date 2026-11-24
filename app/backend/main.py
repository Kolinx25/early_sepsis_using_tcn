"""SepsiGuard FastAPI application.

Endpoints (see docs/api_documentation.md):
    POST /api/patients/{pid}/observations   ingest hourly observations
    GET  /api/overview                      ICU overview counters + recents
    GET  /api/patients                      monitored patients w/ latest risk
    GET  /api/patients/{pid}                patient detail (vitals, risk history)
    GET  /api/patients/{pid}/vitals         recent vitals series (charts)
    GET  /api/predictions/recent            recent prediction updates
    GET  /api/alerts                        alert history
    POST /api/alerts/{id}/ack               acknowledge an alert
    GET  /api/model/performance             genuine evaluation results
    GET  /api/system/health                api/db/model health
    POST /api/replay                        replay controls (start/stop/pause/resume/speed)
    GET  /api/replay/status
    GET  /api/system/logs                   recent system events
    WS   /ws                                live event stream

NOTE: this service is a research prototype; all data are historical
(PhysioNet 2019 test-split replay), clearly labelled as simulated live data.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import db
from .config import RESULTS_METRICS
from .replay import ENGINE
from .service import SERVICE

app = FastAPI(
    title="SepsiGuard",
    description="Early Sepsis Prediction Using Assisted Intelligence for "
                "Clinical Decision Support — research prototype. Historical "
                "ICU replay data (PhysioNet/CinC 2019) presented as simulated "
                "live data.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ObservationIn(BaseModel):
    iculos: int
    channels: dict[str, float | None]


class ReplayCommand(BaseModel):
    action: str  # start | stop | pause | resume | speed
    speed: float | None = None
    n_patients: int | None = None


@app.on_event("startup")
async def startup():
    db.init_pool()
    SERVICE.broadcaster.loop = asyncio.get_event_loop()
    db.log_event("info", f"backend started, model={SERVICE.predictor.version}")


# ------------------------------------------------------------------ ingestion
@app.post("/api/patients/{pid}/observations")
def ingest(pid: int, obs: ObservationIn | list[ObservationIn]):
    obs = obs if isinstance(obs, list) else [obs]
    out = []
    for o in sorted(obs, key=lambda x: x.iculos):
        out.append(SERVICE.ingest_observation(pid, o.iculos, o.channels, "api"))
    return out


# ------------------------------------------------------------------ dashboard
@app.get("/api/overview")
def overview():
    counts = db.query(
        """SELECT
             COUNT(DISTINCT p.patient_id) AS monitored,
             COUNT(*) FILTER (WHERE pr.risk_band = 'critical') AS critical,
             COUNT(*) FILTER (WHERE pr.risk_band = 'high') AS high
           FROM (SELECT DISTINCT ON (patient_id) patient_id, risk_band
                 FROM predictions WHERE prediction_status='ok'
                 ORDER BY patient_id, iculos DESC) pr
           JOIN patients p ON p.patient_id = pr.patient_id""")[0]
    bands = db.query(
        """SELECT risk_band, COUNT(*) AS n FROM
           (SELECT DISTINCT ON (patient_id) risk_band FROM predictions
            WHERE prediction_status='ok' ORDER BY patient_id, iculos DESC) t
           GROUP BY risk_band""")
    band_map = {r["risk_band"]: r["n"] for r in bands}
    recents = db.query(
        """SELECT patient_id, iculos, prob_6h, prob_12h, risk_band,
                  predicted_at, model_version
           FROM predictions WHERE prediction_status='ok'
           ORDER BY predicted_at DESC LIMIT 12""")
    return {
        "monitored_patients": counts["monitored"] or 0,
        "risk_bands": {
            "critical": band_map.get("critical", 0),
            "high": band_map.get("high", 0),
            "elevated": band_map.get("elevated", 0),
            "low": band_map.get("low", 0),
        },
        "recent_predictions": recents,
        "data_source": {
            "kind": "historical_replay",
            "label": "Simulated live data — PhysioNet/CinC 2019 test-split replay",
            "replay": ENGINE.info(),
        },
    }


@app.get("/api/patients")
def patients():
    return db.query(
        """SELECT DISTINCT ON (o.patient_id)
                 o.patient_id, p.first_seen_at, p.last_seen_at,
                 pr.prob_6h, pr.prob_12h, pr.risk_band, pr.iculos,
                 pr.alert_6h, pr.alert_12h, pr.predicted_at, pr.model_version,
                 (SELECT COUNT(*) FROM observations x
                  WHERE x.patient_id = o.patient_id) AS n_observations
           FROM patients p
           JOIN observations o ON o.patient_id = p.patient_id
           LEFT JOIN predictions pr ON pr.patient_id = p.patient_id
                AND pr.prediction_status = 'ok'
           ORDER BY o.patient_id, (COALESCE(pr.iculos, -1)) DESC""")


@app.get("/api/patients/{pid}")
def patient_detail(pid: int):
    p = db.query("SELECT * FROM patients WHERE patient_id=%s", (pid,))
    if not p:
        return {"error": "unknown patient"}
    preds = db.query(
        """SELECT iculos, prob_6h, prob_12h, risk_band, alert_6h, alert_12h,
                  prediction_status, predicted_at
           FROM predictions WHERE patient_id=%s ORDER BY iculos""", (pid,))
    last_obs = db.query(
        "SELECT iculos, channels FROM observations WHERE patient_id=%s "
        "ORDER BY iculos DESC LIMIT 24", (pid,))
    missing = {}
    if last_obs:
        channels = last_obs[0]["channels"]
        missing = {k: (v is None) for k, v in channels.items()}
    return {"patient": p[0], "risk_history": preds,
            "recent_observations": last_obs, "missingness": missing}


@app.get("/api/patients/{pid}/vitals")
def patient_vitals(pid: int, hours: int = 36):
    rows = db.query(
        "SELECT iculos, channels FROM observations WHERE patient_id=%s "
        "ORDER BY iculos DESC LIMIT %s", (pid, hours))
    rows = list(reversed(rows))
    out = []
    for r in rows:
        out.append({"iculos": r["iculos"], **r["channels"]})
    return out


@app.get("/api/predictions/recent")
def recent_predictions(limit: int = 50):
    return db.query(
        """SELECT patient_id, iculos, prob_6h, prob_12h, risk_band,
                  alert_6h, alert_12h, prediction_status, predicted_at, model_version
           FROM predictions ORDER BY predicted_at DESC LIMIT %s""", (limit,))


# ---------------------------------------------------------------------- alerts
@app.get("/api/alerts")
def alerts(only_unacknowledged: bool = False):
    q = """SELECT a.*, p.patient_id AS pid FROM alerts a
           JOIN patients p ON p.patient_id = a.patient_id"""
    if only_unacknowledged:
        return db.query(q + " WHERE a.acknowledged = FALSE ORDER BY a.created_at DESC")
    return db.query(q + " ORDER BY a.created_at DESC")


@app.post("/api/alerts/{alert_id}/ack")
def ack_alert(alert_id: int):
    db.execute("UPDATE alerts SET acknowledged = TRUE WHERE id = %s", (alert_id,))
    return {"ok": True}


# --------------------------------------------------------------- model metrics
@app.get("/api/model/performance")
def model_performance():
    path = RESULTS_METRICS / "final_evaluation.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"error": "evaluation results not found"}


# ----------------------------------------------------------------- system admin
@app.get("/api/system/health")
def health():
    return {
        "api": "healthy",
        "database": "healthy" if db.healthy() else "unavailable",
        "model_version": SERVICE.predictor.version,
        "model_loaded": True,
        "replay": ENGINE.info(),
        "timestamp": db.query("SELECT now() AS ts")[0]["ts"] if db.healthy() else None,
    }


@app.get("/api/system/logs")
def system_logs(limit: int = 50):
    return db.query("SELECT * FROM system_events ORDER BY created_at DESC LIMIT %s",
                    (limit,))


# ------------------------------------------------------------------- replay api
@app.post("/api/replay")
async def replay_command(cmd: ReplayCommand):
    if cmd.action == "start":
        out = ENGINE.start(speed=cmd.speed, n_patients=cmd.n_patients or 25)
        if out["status"] == "running":
            ENGINE._task = asyncio.get_running_loop().create_task(
                ENGINE._run(ENGINE.session_id))
        return out
    if cmd.action == "stop":
        return ENGINE.stop()
    if cmd.action == "pause":
        return ENGINE.pause()
    if cmd.action == "resume":
        return ENGINE.resume()
    if cmd.action == "speed":
        return ENGINE.set_speed(cmd.speed)
    return {"error": f"unknown action {cmd.action}"}


@app.get("/api/replay/status")
def replay_status():
    return ENGINE.info()


# ------------------------------------------------------------------ websockets
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    SERVICE.broadcaster.attach(ws)
    try:
        while True:
            # keep alive; ignore inbound text (e.g. pings)
            msg = await ws.receive_text()
            if msg == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        SERVICE.broadcaster.detach(ws)
    except Exception:
        SERVICE.broadcaster.detach(ws)