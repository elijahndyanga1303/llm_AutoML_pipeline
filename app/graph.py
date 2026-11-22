"""The LangGraph workflow:

START -> profile -> understand -> validate --ok--> plots -> train -> report -> END
                        ^____________|retry (LLM output failed validation)
"""
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from .nodes.plots import plots_node
from .nodes.profile import profile_node
from .nodes.report import report_node
from .nodes.train import train_node
from .nodes.understand import route_after_validate, understand_node, validate_node
from .store import write_status


class State(TypedDict, total=False):
    # inputs
    run_id: str
    data_path: str
    algo: str
    allowed_plots: list
    target: str            # optional user override
    test_size: float
    # produced by nodes
    profile: dict
    understanding: dict
    attempts: int
    errors: list
    plots: list
    skipped_plots: list
    model_plots: list
    metrics: dict
    model_path: str
    task_type: str
    notes: list
    narrative: str
    report_path: str


def _tracked(stage: str, fn):
    """Write the current stage to status.json so n8n/clients can show progress."""
    def wrapper(state: State) -> dict[str, Any]:
        write_status(state["run_id"], stage=stage)
        return fn(state)
    return wrapper


def build_graph():
    g = StateGraph(State)
    g.add_node("profile", _tracked("profiling data", profile_node))
    g.add_node("understand", _tracked("LLM analysing data", understand_node))
    g.add_node("validate", _tracked("validating LLM output", validate_node))
    g.add_node("plots", _tracked("rendering plots", plots_node))
    g.add_node("train", _tracked("training model", train_node))
    g.add_node("report", _tracked("writing report", report_node))

    g.add_edge(START, "profile")
    g.add_edge("profile", "understand")
    g.add_edge("understand", "validate")
    g.add_conditional_edges("validate", route_after_validate,
                            {"retry": "understand", "ok": "plots"})
    g.add_edge("plots", "train")
    g.add_edge("train", "report")
    g.add_edge("report", END)
    return g.compile()


graph = build_graph()
