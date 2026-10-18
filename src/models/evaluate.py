"""Metrics, decision-threshold selection and latency measurement."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Probability cut-off that maximises F1. Must be chosen on out-of-fold predictions from the
    training years only, never on the test set."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "pr_auc": float(average_precision_score(y_true, proba)) if both_classes else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else float("nan"),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier": float(brier_score_loss(y_true, proba)),
        "prevalence": float(y_true.mean()),
        "threshold": float(threshold),
    }


def persistence_proba(df: pd.DataFrame, window: int) -> np.ndarray:
    """Score a lap by the mean pace loss of the previous `window` retained laps of the same stint.

    This is the "tyres were already slow, so they still are" baseline. The label is a trailing mean of pace
    loss, so a model must beat this to show it adds more than the label's own recent history. The baseline is
    generous: pace loss uses the stint's best laps as reference, which a live service would not know yet.
    Laps without history score 0. The score is squashed into [0, 1) because only its ranking matters.
    """
    order = df.sort_values(["session_key", "driver_number", "stint_number", "lap_number"])
    keys = [order["session_key"], order["driver_number"], order["stint_number"]]
    previous = order.groupby(keys)["pace_loss_s"].shift(1)
    rolled = previous.groupby(keys).rolling(window, min_periods=1).mean().reset_index(level=[0, 1, 2], drop=True)
    score = rolled.reindex(df.index).fillna(0.0).clip(lower=0.0).to_numpy()
    return 1.0 - np.exp(-score)


def single_row_latency_ms(predict_fn: Any, rows: pd.DataFrame, n: int) -> dict[str, float]:
    """Wall-clock latency of predicting ONE row at a time, which is what the API does."""
    n = min(n, len(rows))
    timings = []
    for i in range(n):
        start = time.perf_counter()
        predict_fn(rows.iloc[[i]])
        timings.append((time.perf_counter() - start) * 1000.0)
    return {"latency_ms_p50": float(np.percentile(timings, 50)), "latency_ms_p95": float(np.percentile(timings, 95))}
