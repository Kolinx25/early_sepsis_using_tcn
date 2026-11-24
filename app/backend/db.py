"""PostgreSQL access layer (psycopg2, threaded connection pool)."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from .config import DATABASE_URL, PROJECT_ROOT

_pool: ThreadedConnectionPool | None = None
_lock = threading.Lock()


def init_pool(minconn: int = 2, maxconn: int = 10) -> ThreadedConnectionPool:
    global _pool
    with _lock:
        if _pool is None:
            _pool = ThreadedConnectionPool(minconn, maxconn, DATABASE_URL)
            apply_schema(_pool)
    return _pool


def apply_schema(pool: ThreadedConnectionPool | None = None) -> None:
    pool = pool or _pool
    schema = Path(PROJECT_ROOT / "database" / "schema.sql").read_text()
    with conn() as c:
        with c.cursor() as cur:
            cur.execute(schema)
        c.commit()


@contextmanager
def conn():
    if _pool is None:
        init_pool()
    c = _pool.getconn()
    try:
        yield c
    finally:
        _pool.putconn(c)


@contextmanager
def cursor(dict_rows: bool = True):
    with conn() as c:
        with c.cursor(cursor_factory=RealDictCursor if dict_rows else None) as cur:
            yield cur, c


def execute(sql: str, params: tuple = ()) -> None:
    with cursor(False) as (cur, c):
        cur.execute(sql, params)
        c.commit()


def query(sql: str, params: tuple = ()) -> list[dict]:
    with cursor() as (cur, _):
        cur.execute(sql, params)
        return cur.fetchall()


def log_event(level: str, message: str) -> None:
    try:
        execute(
            "INSERT INTO system_events (level, message) VALUES (%s, %s)",
            (level, message),
        )
    except Exception:
        pass


def healthy() -> bool:
    try:
        rows = query("SELECT 1 AS ok")
        return bool(rows)
    except Exception:
        return False


def as_json(rows: list[dict]) -> str:
    return json.dumps(rows, default=str)