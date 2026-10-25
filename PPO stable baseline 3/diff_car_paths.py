"""Shared locations for separate car training runs and their checkpoints."""
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from zoneinfo import ZoneInfo


LOG_ROOT = Path(__file__).resolve().parent.parent / "logs" / "diff_car_balance"
RUN_TIMEZONE = ZoneInfo("America/Los_Angeles")


def create_run_directory(root: Path = LOG_ROOT) -> Path:
    """Use a readable start time plus an atomically allocated unique suffix."""
    run_name = datetime.now(RUN_TIMEZONE).strftime("%Y-%m-%d_%H-%M-%S")
    root.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=f"{run_name}_", dir=root))


def find_latest_checkpoint(root: Path = LOG_ROOT) -> Path:
    """Find the newest checkpoint across run folders, including legacy flat files."""
    checkpoints = list(root.rglob("model*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No model checkpoints found in {root}")
    return max(checkpoints, key=lambda path: (path.stat().st_mtime_ns, path.name == "model.pt"))
