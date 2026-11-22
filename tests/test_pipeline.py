"""Offline smoke test: runs every node EXCEPT the LLM ones, using a hand-written 'understanding'.

    python -m pytest tests -q        (or)        python -m tests.test_pipeline
"""
import os
import sys
import tempfile
import types
import uuid

try:  
    import pydantic 
except ImportError:
    m = types.ModuleType("pydantic")
    m.BaseModel = type("BaseModel", (), {})
    m.Field = lambda *a, **k: None
    sys.modules["pydantic"] = m

os.environ.setdefault("RUNS_DIR", tempfile.mkdtemp())

from app.nodes.plots import plots_node 
from app.nodes.profile import profile_node 
from app.nodes.report import report_node 
from app.nodes.train import train_node 
from app.nodes.understand import validate_node 

CSV = os.path.join(os.path.dirname(__file__), "..", "sample_data", "customers.csv")

FAKE_LLM_OUTPUT = {
    "summary": "Telecom customers with churn flag.",
    "suggested_target": "churn",
    "task_type": "classification",
    "drop_columns": ["customer_id", "does_not_exist"],  # invalid name must be sanitized
    "warnings": [],
    "plots": [
        {"kind": "hist", "x": "monthly_charges", "y": None, "reason": "spend distribution"},
        {"kind": "count", "x": "contract", "y": None, "reason": "contract mix"},
        {"kind": "box", "x": "contract", "y": "monthly_charges", "reason": "spend by contract"},
        {"kind": "scatter", "x": "tenure_months", "y": "monthly_charges", "reason": "tenure vs spend"},
        {"kind": "bar", "x": "contract", "y": "support_calls", "reason": "calls by contract"},
        {"kind": "heatmap_corr", "x": None, "y": None, "reason": "correlations"},
        {"kind": "hist", "x": "ghost_column", "y": None, "reason": "should be dropped"},
    ],
}


def _run(algo: str, target: str = None, understanding=None):
    state = {"run_id": uuid.uuid4().hex, "data_path": CSV, "algo": algo,
             "allowed_plots": ["hist", "count", "box", "scatter", "bar", "heatmap_corr"]}
    if target:
        state["target"] = target
    state.update(profile_node(state))
    state["understanding"] = understanding or FAKE_LLM_OUTPUT
    state["attempts"] = 99  # pretend retries are exhausted -> sanitize path
    state.update(validate_node(state))
    state.update(plots_node(state))
    state.update(train_node(state))
    state.update(report_node(state))
    return state


def test_classification_random_forest():
    s = _run("random_forest")
    assert s["task_type"] == "classification"
    assert s["metrics"]["test"]["accuracy"] > 0.5
    assert "customer_id" not in s["metrics"]["features_used"]
    assert len(s["plots"]) == 6 and not s["skipped_plots"]
    assert os.path.exists(s["report_path"])


def test_regression_ridge():
    u = dict(FAKE_LLM_OUTPUT, suggested_target="monthly_charges", task_type="regression",
             drop_columns=["customer_id", "total_charges"], plots=[])
    s = _run("ridge", target="monthly_charges", understanding=u)
    assert s["task_type"] == "regression"
    assert "r2" in s["metrics"]["test"]


def test_incompatible_algo_gives_clear_error():
    try:
        _run("linear_regression")
    except ValueError as e:
        assert "does not support classification" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_classification_random_forest()
    test_regression_ridge()
    test_incompatible_algo_gives_clear_error()
    print("all tests passed")
