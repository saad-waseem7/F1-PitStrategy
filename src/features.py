"""Single source of truth for turning raw feature rows into model input.

Training, evaluation and the serving API all go through FeaturePreparer, which removes
train/serve skew: column order, dtypes and the compound vocabulary are enforced in one place.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import COMPOUNDS, FEATURE_COLUMNS

NUMERIC_COLUMNS: tuple[str, ...] = tuple(c for c in FEATURE_COLUMNS if c != "compound")
WET_COMPOUNDS: tuple[str, ...] = ("INTERMEDIATE", "WET")


class FeaturePreparer(BaseEstimator, TransformerMixin):
    """Select FEATURE_COLUMNS in fixed order, cast numerics to float64, and encode `compound`
    as a categorical with a fixed vocabulary. Extra columns are dropped. Missing columns,
    unknown compounds and wet-weather compounds raise ValueError instead of silently
    predicting garbage: the models are trained on dry races only."""

    def fit(self, X: pd.DataFrame, y=None) -> FeaturePreparer:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FEATURE_COLUMNS if c not in X.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        out = X.loc[:, list(FEATURE_COLUMNS)].copy()
        compound = out["compound"].astype(str).str.upper()
        unknown = sorted(set(compound) - set(COMPOUNDS))
        if unknown:
            raise ValueError(f"Unknown compound(s) {unknown}; expected one of {list(COMPOUNDS)}")
        wet = sorted(set(compound) & set(WET_COMPOUNDS))
        if wet:
            raise ValueError(f"Wet-weather compound(s) {wet} are out of scope: the model supports dry races only")
        out["compound"] = pd.Categorical(compound, categories=list(COMPOUNDS))
        out[list(NUMERIC_COLUMNS)] = out[list(NUMERIC_COLUMNS)].astype("float64")
        return out
