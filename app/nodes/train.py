"""Node 5: deterministic training. The LLM has no say here beyond which columns to drop."""
import json

import joblib
import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.dummy import DummyClassifier, DummyRegressor  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, classification_report, confusion_matrix, f1_score,
    mean_absolute_error, mean_squared_error, precision_score, r2_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler  # noqa: E402

from ..registry import build_estimator  # noqa: E402
from ..store import run_dir  # noqa: E402
from .profile import load_dataframe  # noqa: E402


def resolve_task(y: pd.Series, suggested: str) -> str:
    """Trust the LLM's task_type unless the target's dtype/cardinality clearly says otherwise."""
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    n = y.nunique(dropna=True)
    if n <= 2:
        return "classification"
    if pd.api.types.is_float_dtype(y) and n > 20:
        return "regression"
    if n > 50:
        return "regression"
    return suggested if suggested in ("classification", "regression") else (
        "classification" if n <= 20 else "regression"
    )


def _prepare_features(X: pd.DataFrame):
    notes = []
    for c in list(X.columns):
        s = X[c]
        if pd.api.types.is_bool_dtype(s):
            X[c] = s.astype(int)
        elif s.nunique(dropna=True) <= 1:
            X = X.drop(columns=c)
            notes.append(f"Dropped constant column '{c}'.")
        elif not pd.api.types.is_numeric_dtype(s) and s.nunique() > max(50, 0.5 * len(X)):
            X = X.drop(columns=c)
            notes.append(f"Dropped high-cardinality text/ID-like column '{c}'.")
    return X, notes


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = [c for c in X.columns if c not in num]
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if cat:
        parts.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", max_categories=30))]), cat))
    if not parts:
        raise ValueError("No usable feature columns left after cleaning.")
    return ColumnTransformer(parts)


def _importance(pipe, top=15):
    est, pre = pipe.named_steps["model"], pipe.named_steps["prep"]
    try:
        names = pre.get_feature_names_out()
    except Exception:
        return []
    if hasattr(est, "feature_importances_"):
        vals = np.asarray(est.feature_importances_)
    elif hasattr(est, "coef_"):
        c = np.abs(np.asarray(est.coef_))
        vals = c.mean(axis=0) if c.ndim > 1 else c
    else:
        return []
    if len(vals) != len(names):
        return []
    idx = np.argsort(vals)[::-1][:top]
    clean = lambda n: str(n).replace("num__", "").replace("cat__", "")  # noqa: E731
    return [{"feature": clean(names[i]), "importance": float(vals[i])} for i in idx]


def _save(fig, path):
    fig.savefig(path, dpi=110, bbox_inches="tight")


def _model_plots(out, task, importances, y_te, pred, class_names):
    plots = []
    if importances:
        fig = Figure(figsize=(7.5, 5)); ax = fig.subplots()
        d = importances[::-1]
        ax.barh([i["feature"] for i in d], [i["importance"] for i in d])
        ax.set_title("Top feature importances")
        _save(fig, out / "model_feature_importance.png")
        plots.append({"name": "model_feature_importance.png", "kind": "feature_importance",
                      "reason": "Which features the model relies on most"})
    fig = Figure(figsize=(6.5, 5)); ax = fig.subplots()
    if task == "classification":
        cm = confusion_matrix(y_te, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion matrix (test set)")
        name, kind = "model_confusion_matrix.png", "confusion_matrix"
    else:
        ax.scatter(y_te, pred, alpha=0.5)
        lo, hi = float(min(np.min(y_te), np.min(pred))), float(max(np.max(y_te), np.max(pred)))
        ax.plot([lo, hi], [lo, hi], "r--")
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted"); ax.set_title("Predicted vs actual (test set)")
        name, kind = "model_pred_vs_actual.png", "pred_vs_actual"
    _save(fig, out / name)
    plots.append({"name": name, "kind": kind, "reason": "How predictions compare to the truth"})
    return plots


def train_node(state: dict) -> dict:
    out = run_dir(state["run_id"]) / "plots"
    out.mkdir(exist_ok=True)

    df = load_dataframe(state["data_path"])
    df.columns = [str(c).strip() for c in df.columns]
    target, u = state["target"], state["understanding"]

    df = df.dropna(subset=[target])
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} rows have a target value - need at least 30 to train.")

    drop = [c for c in u["drop_columns"] if c in df.columns and c != target]
    y_raw = df[target]
    X = df.drop(columns=[target] + drop).copy()
    X, notes = _prepare_features(X)
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    X[cat_cols] = X[cat_cols].astype(object).where(X[cat_cols].notna(), np.nan)

    task = resolve_task(y_raw, u["task_type"])
    estimator = build_estimator(state["algo"], task)  # raises a clear error if incompatible

    le, class_names = None, None
    if task == "classification":
        le = LabelEncoder()
        y = le.fit_transform(y_raw.astype(str))
        class_names = [str(c) for c in le.classes_]
        counts = np.bincount(y)
        if len(counts) < 2:
            raise ValueError("The target has only one class.")
        if counts.min() < 2:
            raise ValueError("Every class needs at least 2 rows (a class has a single sample).")
        stratify = y
    else:
        y = y_raw.astype(float).to_numpy()
        stratify = None

    test_size = float(state.get("test_size") or 0.2)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify)

    pipe = Pipeline([("prep", _preprocessor(X)), ("model", estimator)])

    # ---- cross-validation on the training split
    scoring = ({"accuracy": "accuracy", "f1_weighted": "f1_weighted"} if task == "classification"
               else {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error",
                     "neg_mae": "neg_mean_absolute_error"})
    cv_res = None
    try:
        if task == "classification":
            k = min(5, int(np.bincount(y_tr).min()))
            cv = StratifiedKFold(k, shuffle=True, random_state=42) if k >= 2 else None
        else:
            cv = 5
        if cv is not None:
            raw = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=scoring, n_jobs=1)
            cv_res = {}
            for m in scoring:
                v = raw[f"test_{m}"]
                if m.startswith("neg_"):
                    v, m = -v, m[4:]
                cv_res[m] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
    except Exception as e:
        notes.append(f"Cross-validation skipped: {e}")

    # ---- final fit + test metrics
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    dummy_X_tr, dummy_X_te = np.zeros((len(X_tr), 1)), np.zeros((len(X_te), 1))

    if task == "classification":
        test = {
            "accuracy": accuracy_score(y_te, pred),
            "f1_weighted": f1_score(y_te, pred, average="weighted", zero_division=0),
            "precision_weighted": precision_score(y_te, pred, average="weighted", zero_division=0),
            "recall_weighted": recall_score(y_te, pred, average="weighted", zero_division=0),
        }
        try:
            proba = pipe.predict_proba(X_te)
            test["roc_auc"] = (roc_auc_score(y_te, proba[:, 1]) if proba.shape[1] == 2 else
                               roc_auc_score(y_te, proba, multi_class="ovr", average="weighted",
                                             labels=list(range(proba.shape[1]))))
        except Exception:
            pass
        base = DummyClassifier(strategy="most_frequent").fit(dummy_X_tr, y_tr).predict(dummy_X_te)
        baseline = {"accuracy": accuracy_score(y_te, base),
                    "f1_weighted": f1_score(y_te, base, average="weighted", zero_division=0)}
        extra = {"classification_report": classification_report(
            y_te, pred, target_names=class_names, output_dict=True, zero_division=0),
            "classes": class_names}
    else:
        test = {"r2": r2_score(y_te, pred),
                "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
                "mae": mean_absolute_error(y_te, pred)}
        base = DummyRegressor().fit(dummy_X_tr, y_tr).predict(dummy_X_te)
        baseline = {"r2": r2_score(y_te, base),
                    "rmse": float(np.sqrt(mean_squared_error(y_te, base))),
                    "mae": mean_absolute_error(y_te, base)}
        extra = {}

    importances = _importance(pipe)
    model_plots = _model_plots(out, task, importances, y_te, pred, class_names)

    model_path = run_dir(state["run_id"]) / "model.joblib"
    joblib.dump({"pipeline": pipe, "label_encoder": le, "task": task, "target": target,
                 "features": list(X.columns), "algo": state["algo"]}, model_path)

    metrics = {
        "task_type": task,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "features_used": list(X.columns),
        "cv": cv_res,
        "test": {k: float(v) for k, v in test.items()},
        "baseline": {k: float(v) for k, v in baseline.items()},
        "feature_importance": importances,
        **extra,
    }
    (run_dir(state["run_id"]) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return {"metrics": metrics, "model_path": str(model_path), "task_type": task,
            "model_plots": model_plots, "notes": notes}
