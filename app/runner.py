"""Executes one run in the background and reports progress/results to status.json."""
import logging
import traceback
from typing import Optional

import httpx

from .graph import graph
from .store import read_status, write_status

log = logging.getLogger("automl")


def _result(run_id: str, out: dict) -> dict:
    u = out["understanding"]
    link = lambda kind, items: [  # noqa: E731
        {**p, "url": f"/runs/{run_id}/plots/{p['name']}"} for p in items]
    return {
        "algorithm": out["algo"],
        "target": out["target"],
        "task_type": out["task_type"],
        "dataset_summary": u["summary"],
        "warnings": u["warnings"],
        "dropped_columns": u["drop_columns"],
        "preprocessing_notes": out.get("notes", []),
        "metrics": out["metrics"],
        "narrative": out["narrative"],
        "plots": link("eda", out.get("plots", [])),
        "model_plots": link("model", out.get("model_plots", [])),
        "skipped_plots": out.get("skipped_plots", []),
        "report_url": f"/runs/{run_id}/report",
        "model_url": f"/runs/{run_id}/model",
    }


def execute_run(run_id: str, data_path: str, algo: str, plots: list,
                target: Optional[str], test_size: float, callback_url: Optional[str]) -> None:
    write_status(run_id, status="running", stage="starting")
    try:
        state = {"run_id": run_id, "data_path": data_path, "algo": algo,
                 "allowed_plots": plots, "test_size": test_size}
        if target:
            state["target"] = target
        out = graph.invoke(state)
        write_status(run_id, status="done", stage="finished", result=_result(run_id, out))
    except Exception as e:
        log.error("run %s failed:\n%s", run_id, traceback.format_exc())
        write_status(run_id, status="failed", stage="failed", error=f"{type(e).__name__}: {e}")

    if callback_url:  # optional: tell n8n (or anything) that the run is over
        try:
            httpx.post(callback_url, json=read_status(run_id), timeout=15)
        except Exception as e:
            log.warning("callback to %s failed: %s", callback_url, e)
