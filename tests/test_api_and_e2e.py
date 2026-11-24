"""Integration tests: API, database persistence, replay engine, WebSockets.
Skipped automatically when no backend/database is reachable (e.g. unit-only
environments)."""
import json
import os
import time
import urllib.request

import pytest

API = os.environ.get("SEPSIGUARD_API", "http://127.0.0.1:8000")


def api_available() -> bool:
    try:
        urllib.request.urlopen(f"{API}/api/system/health", timeout=3)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not api_available(), reason="backend not running")


def call(path, data=None, method=None):
    req = urllib.request.Request(
        f"{API}{path}",
        data=json.dumps(data).encode() if data is not None else None,
        headers={"content-type": "application/json"},
        method=method or ("POST" if data is not None else "GET"))
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def db_reachable() -> bool:
    try:
        import psycopg2
        from app.backend.config import DATABASE_URL
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
        conn.close()
        return True
    except Exception:
        return False


# ------------------------------------------------------------- database layer
@pytest.mark.skipif(not db_reachable(), reason="database not reachable")
def test_database_schema_and_prediction_persistence():
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from app.backend.config import DATABASE_URL
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""SELECT table_name FROM information_schema.tables
                   WHERE table_schema='public'""")
    tables = {r["table_name"] for r in cur.fetchall()}
    for t in ("patients", "observations", "predictions", "alerts",
              "replay_sessions", "system_events"):
        assert t in tables, f"missing table {t}"
    cur.execute("SELECT count(*) AS n FROM predictions WHERE prediction_status='ok'")
    assert cur.fetchone()["n"] > 0, "no persisted predictions found"
    cur.execute("""SELECT count(*) AS n FROM predictions
                   WHERE prediction_status='ok' AND prob_6h IS NULL""")
    assert cur.fetchone()["n"] == 0, "ok-prediction without probability"
    conn.close()


# ----------------------------------------------------------------- API surface
def test_health_endpoint():
    h = call("/api/system/health")
    assert h["api"] == "healthy"
    assert h["database"] == "healthy"
    assert h["model_version"].startswith("sepsiguard")


def test_ingest_pending_then_prediction():
    pid = int(time.time()) % 1_000_000 + 9_000_000
    for h in range(1, 13):
        out = call(f"/api/patients/{pid}/observations",
                   {"iculos": h, "channels": {"HR": 95, "SBP": 110, "Temp": 37.1}})[0]
    assert out["prediction_status"] == "ok"
    assert 0.0 <= out["prob_6h"] <= 1.0
    assert 0.0 <= out["prob_12h"] <= 1.0
    assert out["patient_id"] == pid
    assert out["model_version"].startswith("sepsiguard")


def test_pending_history_never_zero_risk():
    pid = int(time.time()) % 1_000_000 + 9_500_000
    out = call(f"/api/patients/{pid}/observations",
               {"iculos": 1, "channels": {"HR": 90}})[0]
    assert out["prediction_status"] == "pending_history"
    assert out.get("prob_6h") is None and out.get("prob_12h") is None


def test_overview_and_alerts_endpoints():
    ov = call("/api/overview")
    assert "monitored_patients" in ov
    assert "risk_bands" in ov
    assert ov["data_source"]["kind"] == "historical_replay"
    alerts = call("/api/alerts")
    assert isinstance(alerts, list)


def test_model_performance_serves_genuine_metrics():
    m = call("/api/model/performance")
    assert "deployed_variant" in m
    assert 0.5 < m["test_y12"]["auroc"] < 1.0
    assert 0 < m["test_y12"]["auprc"] < 1.0


# ------------------------------------------------------------ replay + socket
def test_replay_engine_and_websocket():
    import asyncio
    import websockets

    async def scenario():
        call("/api/replay", {"action": "stop"}) if call("/api/replay/status")["status"] == "running" else None
        async with websockets.connect(f"{API.replace('http', 'ws')}/ws") as ws:
            call("/api/replay", {"action": "start", "n_patients": 6, "speed": 40})
            got_prediction = got_alert = False
            deadline = time.time() + 90
            while time.time() < deadline and not (got_prediction):
                try:
                    msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                except asyncio.TimeoutError:
                    break
                if msg.get("type") == "prediction" and msg.get("prediction_status") == "ok":
                    got_prediction = True
                    assert "prob_6h" in msg and "prob_12h" in msg
                    assert "patient_id" in msg and "model_version" in msg
                if msg.get("type") == "observation":
                    assert msg.get("prediction_status") in ("pending_history", "ok")
            call("/api/replay", {"action": "stop"})
            return got_prediction, got_alert

    ok, _ = asyncio.run(scenario())
    assert ok, "expected at least one live prediction event over WebSocket"


# --------------------------------------------------- MANDATORY ACCEPTANCE TEST
def test_end_to_end_acceptance():
    """Start application -> start historical ICU replay -> receive observation
    -> update patient features -> generate predictions -> save predictions
    -> push WebSocket update -> display updated risk on the dashboard."""
    import asyncio
    import websockets

    async def scenario():
        call("/api/replay", {"action": "stop"})
        async with websockets.connect(f"{API.replace('http', 'ws')}/ws") as ws:
            call("/api/replay", {"action": "start", "n_patients": 8, "speed": 60})
            deadline = time.time() + 120
            events = []
            while time.time() < deadline:
                try:
                    events.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=20)))
                except asyncio.TimeoutError:
                    break
                if len([e for e in events if e.get("type") == "prediction" and
                        e.get("prediction_status") == "ok"]) >= 5:
                    break
            return events

    events = asyncio.run(scenario())
    call("/api/replay", {"action": "stop"})
    preds = [e for e in events if e.get("type") == "prediction"
             and e.get("prediction_status") == "ok"]
    assert len(preds) >= 5, f"expected >=5 prediction events, got {len(preds)}"
    # 2. persisted in PostgreSQL
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from app.backend.config import DATABASE_URL
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor(cursor_factory=RealDictCursor)
    pids = [e["patient_id"] for e in preds]
    cur.execute("SELECT count(*) AS n FROM predictions WHERE prediction_status='ok'")
    assert cur.fetchone()["n"] >= len(preds)
    # 3. risk visible on dashboard endpoints
    ov = call("/api/overview")
    assert ov["monitored_patients"] >= 1
    pat = call("/api/patients")
    assert any(p["patient_id"] in pids for p in pat)
    conn.close()