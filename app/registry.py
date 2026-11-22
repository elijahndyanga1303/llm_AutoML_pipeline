"""Maps the algorithm name chosen by the user to a scikit-learn / XGBoost class."""
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ALGOS: dict = {
    "random_forest": {
        "classification": (RandomForestClassifier, dict(n_estimators=300, n_jobs=-1, random_state=42)),
        "regression": (RandomForestRegressor, dict(n_estimators=300, n_jobs=-1, random_state=42)),
    },
    "gradient_boosting": {
        "classification": (GradientBoostingClassifier, dict(random_state=42)),
        "regression": (GradientBoostingRegressor, dict(random_state=42)),
    },
    "logistic_regression": {
        "classification": (LogisticRegression, dict(max_iter=2000)),
    },
    "linear_regression": {
        "regression": (LinearRegression, dict()),
    },
    "ridge": {
        "regression": (Ridge, dict(alpha=1.0)),
    },
    "decision_tree": {
        "classification": (DecisionTreeClassifier, dict(max_depth=8, random_state=42)),
        "regression": (DecisionTreeRegressor, dict(max_depth=8, random_state=42)),
    },
    "knn": {
        "classification": (KNeighborsClassifier, dict(n_neighbors=5)),
        "regression": (KNeighborsRegressor, dict(n_neighbors=5)),
    },
}

# XGBoost is optional so the rest of the app still works if it is not installed
try:
    from xgboost import XGBClassifier, XGBRegressor

    ALGOS["xgboost"] = {
        "classification": (
            XGBClassifier,
            dict(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                 n_jobs=-1, random_state=42, eval_metric="logloss"),
        ),
        "regression": (
            XGBRegressor,
            dict(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                 n_jobs=-1, random_state=42),
        ),
    }
except ImportError:  # pragma: no cover
    pass


def list_algorithms() -> dict:
    return {name: sorted(tasks) for name, tasks in ALGOS.items()}


def build_estimator(algo: str, task: str):
    if algo not in ALGOS:
        raise ValueError(f"Unknown algorithm '{algo}'. Available: {sorted(ALGOS)}")
    if task not in ALGOS[algo]:
        ok = [a for a, t in ALGOS.items() if task in t]
        raise ValueError(
            f"Algorithm '{algo}' does not support {task}. "
            f"The target looks like a {task} problem - choose one of: {ok}"
        )
    cls, params = ALGOS[algo][task]
    return cls(**params)
