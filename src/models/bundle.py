"""The deployable artifact: regressor + classifier + decision threshold in one MLflow pyfunc.

Serving code only needs:
    model = mlflow.pyfunc.load_model("models/champion")
    model.predict(df)   # -> DataFrame[pace_loss_s, pit_probability, pit_recommended]
"""
from __future__ import annotations

from typing import Any

import mlflow.pyfunc
import pandas as pd


class PitStrategyModel(mlflow.pyfunc.PythonModel):
    def __init__(self, regressor: Any, classifier: Any, threshold: float) -> None:
        self.regressor = regressor
        self.classifier = classifier
        self.threshold = float(threshold)

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        loss = self.regressor.predict(X)
        proba = self.classifier.predict_proba(X)[:, 1]
        return pd.DataFrame(
            {
                "pace_loss_s": loss.astype("float64"),
                "pit_probability": proba.astype("float64"),
                "pit_recommended": proba >= self.threshold,
            },
            index=X.index,
        )

    def predict(self, context, model_input, params=None) -> pd.DataFrame:
        return self.predict_frame(pd.DataFrame(model_input))
