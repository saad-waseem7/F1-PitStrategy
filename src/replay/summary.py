"""How the model did on the whole replay sample, overall and by race and tyre compound."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ACTUAL_LOSS = "pace_loss_s"
ACTUAL_FLAG = "pit_recommended"


def _ratio(num: float, den: float) -> float | None:
    return float(num / den) if den > 0 else None


def metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Error and flag metrics for rows that carry actuals and predictions.

    Needs columns pace_loss_s, pit_recommended (the labels) and predicted_pace_loss_s, pit_probability,
    predicted_flag. `bias` is mean(predicted - actual): positive means the model over-predicts pace loss.
    Metrics that need both classes or a non-constant target are None when the data cannot support them.
    """
    n = len(frame)
    if n == 0:
        raise ValueError("no rows to summarise")
    actual = frame[ACTUAL_LOSS].to_numpy(dtype=float)
    pred = frame["predicted_pace_loss_s"].to_numpy(dtype=float)
    err = pred - actual
    label = frame[ACTUAL_FLAG].to_numpy(dtype=bool)
    flag = frame["predicted_flag"].to_numpy(dtype=bool)
    tp = int(np.sum(label & flag))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    precision = _ratio(tp, int(flag.sum()))
    recall = _ratio(tp, int(label.sum()))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    both_classes = 0 < int(label.sum()) < n
    return {
        "laps": int(n),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias": float(err.mean()),
        "r2": _ratio(ss_tot - float(np.sum(err**2)), ss_tot) if ss_tot > 0 else None,
        "flag_accuracy": float((label == flag).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": float(average_precision_score(label, frame["pit_probability"])) if both_classes else None,
        "prevalence": float(label.mean()),
    }


def summarize(frame: pd.DataFrame) -> dict[str, Any]:
    by_session = [
        {"session_key": int(key), "circuit": str(grp["circuit_short_name"].iloc[0]), "year": int(grp["year"].iloc[0]),
         "metrics": metrics(grp)}
        for key, grp in frame.groupby("session_key", sort=True)
    ]
    by_compound = [{"compound": str(key), "metrics": metrics(grp)} for key, grp in frame.groupby("compound", sort=True)]
    return {"overall": metrics(frame), "by_session": by_session, "by_compound": by_compound}
