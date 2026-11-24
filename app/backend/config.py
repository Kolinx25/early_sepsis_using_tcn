"""SepsiGuard backend configuration (environment-driven)."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://sepsiguard:sepsiguard_research@localhost:5432/sepsiguard",
)
MODEL_VERSION = os.environ.get("MODEL_VERSION", "sepsiguard_v1.0")
REGISTRY_DIR = Path(os.environ.get(
    "MODEL_REGISTRY_DIR", str(PROJECT_ROOT / "artifacts" / "model_registry")))
REPLAY_POOL = Path(os.environ.get(
    "REPLAY_POOL", str(PROJECT_ROOT / "data" / "simulation" / "replay_sessions"
                       / "test_replay_pool.parquet")))
RESULTS_METRICS = PROJECT_ROOT / "results" / "metrics"
DEFAULT_REPLAY_SPEED = float(os.environ.get("REPLAY_SPEED", "2.0"))  # sim hours / s
API_PORT = int(os.environ.get("API_PORT", "8000"))