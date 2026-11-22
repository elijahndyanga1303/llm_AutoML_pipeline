"""Tiny file-based run store: runs/<run_id>/status.json (+ artifacts)."""
import json
import threading
import time
from pathlib import Path
from typing import Optional

from .config import RUNS_DIR

_lock = threading.Lock()


def run_dir(run_id: str) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_status(run_id: str, /,**fields) -> None:
    with _lock:
        p = run_dir(run_id) / "status.json"
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(fields)
        cur["updated_at"] = time.time()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, default=str))
        tmp.replace(p)


def read_status(run_id: str) -> Optional[dict]:
    p = RUNS_DIR / run_id / "status.json"
    if not p.exists():
        return None
    with _lock:
        return json.loads(p.read_text())
