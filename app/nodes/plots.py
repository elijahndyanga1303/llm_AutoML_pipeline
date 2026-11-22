"""Node 4: render the plots the LLM planned, using ONLY whitelisted functions.

The LLM never writes code - it returns JSON specs that are validated and rendered here.
Uses matplotlib's object-oriented Figure API (no pyplot global state) so that several
runs can render concurrently in different threads.
"""
import re

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .. import config  # noqa: E402
from ..store import run_dir  # noqa: E402
from .profile import load_dataframe  # noqa: E402

MAX_CATS = 15


def _fig(w=7.5, h=4.8):
    fig = Figure(figsize=(w, h))
    return fig, fig.subplots()


def _is_num(df, c):
    return pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])


def _need(df, col, kind):
    if not col or col not in df.columns:
        raise ValueError(f"column '{col}' not found")
    if kind == "num" and not _is_num(df, col):
        raise ValueError(f"column '{col}' must be numeric")
    return col


def _top(df, col):
    return df[col].value_counts().head(MAX_CATS).index.tolist()


def hist(df, s, target):
    x = _need(df, s.get("x"), "any")
    if not _is_num(df, x):
        return count(df, s, target)
    fig, ax = _fig()
    sns.histplot(df[x].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution of {x}")
    return fig


def box(df, s, target):
    y = _need(df, s.get("y") or s.get("x"), "num")
    x = s.get("x") if s.get("y") else None
    fig, ax = _fig()
    if x and x in df.columns and x != y:
        cats = _top(df, x)
        sns.boxplot(data=df[df[x].isin(cats)], x=x, y=y, order=cats, ax=ax)
        ax.set_title(f"{y} by {x}")
        ax.tick_params(axis="x", rotation=45)
    else:
        sns.boxplot(y=df[y].dropna(), ax=ax)
        ax.set_title(f"Box plot of {y}")
    return fig


def scatter(df, s, target):
    x, y = _need(df, s.get("x"), "num"), _need(df, s.get("y"), "num")
    d = df[[x, y] + ([target] if target in df.columns and target not in (x, y) else [])].dropna()
    d = d.sample(min(len(d), 5000), random_state=42)
    fig, ax = _fig()
    if target in d.columns and d[target].nunique() <= 10:
        d["_hue"] = d[target].astype(str)
        sns.scatterplot(data=d, x=x, y=y, hue="_hue", alpha=0.6, ax=ax)
        ax.legend(title=target)
    else:
        sns.scatterplot(data=d, x=x, y=y, alpha=0.5, ax=ax)
    ax.set_title(f"{y} vs {x}")
    return fig


def bar(df, s, target):
    x, y = _need(df, s.get("x"), "any"), _need(df, s.get("y"), "num")
    g = df.groupby(x)[y].mean().sort_values(ascending=False).head(MAX_CATS)
    fig, ax = _fig()
    sns.barplot(x=g.values, y=g.index.astype(str), orient="h", ax=ax)
    ax.set_title(f"Mean {y} by {x}")
    ax.set_xlabel(f"mean {y}")
    return fig


def count(df, s, target):
    x = _need(df, s.get("x"), "any")
    if df[x].nunique() > 200:
        raise ValueError(f"column '{x}' has too many distinct values for a count plot")
    cats = _top(df, x)
    fig, ax = _fig()
    sns.countplot(data=df[df[x].isin(cats)], y=x, order=cats, ax=ax)
    ax.set_title(f"Count of {x}")
    return fig


def heatmap_corr(df, s, target):
    num = df.select_dtypes("number").iloc[:, :25]
    if num.shape[1] < 2:
        raise ValueError("need at least 2 numeric columns")
    n = num.shape[1]
    fig, ax = _fig(max(6, n * 0.5), max(5, n * 0.45))
    sns.heatmap(num.corr(), cmap="coolwarm", center=0, annot=n <= 12, fmt=".2f", ax=ax)
    ax.set_title("Correlation matrix")
    return fig


RENDERERS = {
    "hist": hist, "box": box, "scatter": scatter,
    "bar": bar, "count": count, "heatmap_corr": heatmap_corr,
}


def plots_node(state: dict) -> dict:
    df = load_dataframe(state["data_path"])
    df.columns = [str(c).strip() for c in df.columns]
    if len(df) > config.EDA_SAMPLE_ROWS:
        df = df.sample(config.EDA_SAMPLE_ROWS, random_state=42)

    out_dir = run_dir(state["run_id"]) / "plots"
    out_dir.mkdir(exist_ok=True)
    allowed = set(state["allowed_plots"])
    results, skipped = [], []

    for i, spec in enumerate(state["understanding"]["plots"]):
        kind = spec["kind"]
        if kind not in allowed or kind not in RENDERERS:
            skipped.append({"spec": spec, "error": "kind not allowed"})
            continue
        try:
            fig = RENDERERS[kind](df, spec, state.get("target"))
            name = re.sub(r"[^A-Za-z0-9_-]+", "_",
                          f"{i:02d}_{kind}_{spec.get('x') or ''}_{spec.get('y') or ''}")[:80].strip("_") + ".png"
            fig.savefig(out_dir / name, dpi=110, bbox_inches="tight")
            results.append({"name": name, "kind": kind, "x": spec.get("x"),
                            "y": spec.get("y"), "reason": spec.get("reason", "")})
        except Exception as e:  # one bad plot must not kill the run
            skipped.append({"spec": spec, "error": str(e)})
    return {"plots": results, "skipped_plots": skipped}
