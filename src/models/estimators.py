"""Model factories. Every model is an sklearn Pipeline that starts with FeaturePreparer, so it
accepts a raw feature DataFrame (compound as plain strings) at train and serve time."""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import COMPOUNDS
from src.features import FeaturePreparer

ALGORITHMS: tuple[str, ...] = ("random_forest", "lightgbm")

# LightGBM's documented switches for identical results from identical data, params and seed,
# independent of thread count (reproducibility requirement of the course).
LGBM_STABLE: dict[str, Any] = {"deterministic": True, "force_row_wise": True}

LGBM_SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [200, 400, 600],
    "learning_rate": [0.03, 0.05, 0.08],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 40, 80],
    "subsample": [0.7, 0.85, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "reg_lambda": [0.0, 1.0, 5.0],
}


def _group_keys(X: pd.DataFrame, max_age: int) -> pd.DataFrame:
    """(compound, tyre age capped at max_age) per row: the cell a baseline looks up."""
    return pd.DataFrame(
        {"compound": X["compound"].astype(str).to_numpy(), "age": np.minimum(X["tyre_age"].to_numpy(), max_age)}
    )


class GroupMedianBaseline(BaseEstimator, RegressorMixin):
    """Naive reference: median target per (compound, tyre_age capped at max_age).
    Any real model has to beat this to justify its complexity."""

    def __init__(self, max_age: int = 40) -> None:
        self.max_age = max_age

    def _keys(self, X: pd.DataFrame) -> pd.DataFrame:
        return _group_keys(X, self.max_age)

    def fit(self, X: pd.DataFrame, y) -> GroupMedianBaseline:
        keys = self._keys(X)
        keys["y"] = np.asarray(y, dtype=float)
        self.table_ = keys.groupby(["compound", "age"])["y"].median()
        self.global_median_ = float(np.median(keys["y"]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = self._keys(X)
        idx = pd.MultiIndex.from_frame(keys[["compound", "age"]])
        return self.table_.reindex(idx).fillna(self.global_median_).to_numpy()


class GroupRateBaseline(BaseEstimator, ClassifierMixin):
    """Naive reference classifier: share of pit-flagged laps per (compound, tyre_age capped at
    max_age). It uses no pace information at all, so the real classifier must beat it to show
    that lap-time features add value beyond "how old are the tyres"."""

    def __init__(self, max_age: int = 40) -> None:
        self.max_age = max_age

    def fit(self, X: pd.DataFrame, y) -> GroupRateBaseline:
        keys = _group_keys(X, self.max_age)
        keys["y"] = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        self.table_ = keys.groupby(["compound", "age"])["y"].mean()
        self.global_rate_ = float(keys["y"].mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(_group_keys(X, self.max_age)[["compound", "age"]])
        rate = self.table_.reindex(idx).fillna(self.global_rate_).to_numpy()
        return np.column_stack([1.0 - rate, rate])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _forest_pipeline(estimator: Any) -> Pipeline:
    encode = ColumnTransformer(
        [("compound", OneHotEncoder(categories=[list(COMPOUNDS)], handle_unknown="ignore", sparse_output=False), ["compound"])],
        remainder="passthrough",
    )
    return Pipeline([("prep", FeaturePreparer()), ("encode", encode), ("model", estimator)])


def _lgbm_pipeline(estimator: Any) -> Pipeline:
    return Pipeline([("prep", FeaturePreparer()), ("model", estimator)])


def build_regressor(algorithm: str, params: dict[str, Any], seed: int) -> Pipeline:
    if algorithm == "random_forest":
        return _forest_pipeline(RandomForestRegressor(random_state=seed, n_jobs=-1, **params))
    if algorithm == "lightgbm":
        return _lgbm_pipeline(LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, subsample_freq=1, **LGBM_STABLE, **params))
    raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}")


def build_classifier(algorithm: str, rf_params: dict[str, Any], lgbm_params: dict[str, Any], seed: int) -> Pipeline:
    if algorithm == "random_forest":
        return _forest_pipeline(RandomForestClassifier(random_state=seed, n_jobs=-1, **rf_params))
    if algorithm == "lightgbm":
        return _lgbm_pipeline(LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, subsample_freq=1, **LGBM_STABLE, **lgbm_params))
    raise ValueError(f"Unknown algorithm {algorithm!r}; expected one of {ALGORITHMS}")


def sample_lgbm_params(n_trials: int, seed: int) -> list[dict[str, Any]]:
    """Distinct random configurations from LGBM_SEARCH_SPACE (reproducible for a given seed)."""
    rng = random.Random(seed)
    total = int(np.prod([len(v) for v in LGBM_SEARCH_SPACE.values()]))
    n_trials = min(n_trials, total)
    seen: set[tuple] = set()
    trials: list[dict[str, Any]] = []
    while len(trials) < n_trials:
        candidate = {k: rng.choice(v) for k, v in LGBM_SEARCH_SPACE.items()}
        key = tuple(sorted(candidate.items()))
        if key not in seen:
            seen.add(key)
            trials.append(candidate)
    return trials
