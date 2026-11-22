"""Node 1: load the CSV and build a compact profile. The LLM only ever sees this profile."""
import pandas as pd

from .. import config


def load_dataframe(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")


def _col_profile(s: pd.Series) -> dict:
    info = {
        "dtype": str(s.dtype),
        "null_pct": round(float(s.isna().mean()), 4),
        "n_unique": int(s.nunique(dropna=True)),
        "sample_values": [str(v)[:40] for v in s.dropna().head(3).tolist()],
    }
    if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
        d = s.describe()
        info.update(
            min=float(d["min"]), max=float(d["max"]),
            mean=round(float(d["mean"]), 4), std=round(float(d["std"]), 4)
            if pd.notna(d["std"]) else 0.0,
        )
    else:
        top = s.value_counts().head(5)
        info["top_values"] = {str(k)[:40]: int(v) for k, v in top.items()}
    return info


def build_profile(df: pd.DataFrame) -> dict:
    sample = df if len(df) <= config.EDA_SAMPLE_ROWS else df.sample(
        config.EDA_SAMPLE_ROWS, random_state=42
    )
    cols = list(df.columns)[:150]  # keep the prompt bounded on very wide data
    return {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": {str(c): _col_profile(sample[c]) for c in cols},
    }


def profile_node(state: dict) -> dict:
    df = load_dataframe(state["data_path"])
    if df.empty or df.shape[1] < 2:
        raise ValueError("The dataset needs at least 2 columns and 1 row.")
    df.columns = [str(c).strip() for c in df.columns]
    return {"profile": build_profile(df), "attempts": 0, "errors": []}
