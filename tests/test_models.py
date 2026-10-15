from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.models.bundle import PitStrategyModel
from src.models.estimators import (
    GroupMedianBaseline,
    GroupRateBaseline,
    build_classifier,
    build_regressor,
    sample_lgbm_params,
)
from src.models.evaluate import best_f1_threshold, classification_metrics, regression_metrics, single_row_latency_ms


def make_features(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "tyre_age": rng.integers(0, 30, n).astype(float),
            "compound": rng.choice(["SOFT", "MEDIUM", "HARD"], n),
            "stint_number": rng.integers(1, 4, n).astype(float),
            "lap_number": rng.integers(2, 60, n).astype(float),
            "race_progress": rng.uniform(0, 1, n),
            "track_temperature": rng.uniform(25, 50, n),
            "air_temperature": rng.uniform(15, 35, n),
            "humidity": rng.uniform(20, 80, n),
            "rainfall": np.zeros(n),
            "wind_speed": rng.uniform(0, 6, n),
            "track_temp_delta": rng.normal(0, 1, n),
            "prev_lap_delta": rng.normal(0, 0.3, n),
            "pace_decay_slope": rng.normal(0.05, 0.05, n),
        }
    )
    df.loc[rng.random(n) < 0.2, ["prev_lap_delta", "pace_decay_slope"]] = np.nan
    y = pd.Series(0.06 * df["tyre_age"] * np.where(df["compound"] == "SOFT", 1.6, 1.0) + rng.normal(0, 0.1, n))
    return df[list(FEATURE_COLUMNS)], y, (y > 1.0).astype(int)


def test_group_median_baseline_predicts_group_medians_and_falls_back():
    X = pd.DataFrame({"compound": ["SOFT"] * 3 + ["HARD"] * 3, "tyre_age": [1, 1, 1, 2, 2, 2]})
    model = GroupMedianBaseline().fit(X, [1.0, 2.0, 3.0, 10.0, 10.0, 10.0])
    out = model.predict(pd.DataFrame({"compound": ["SOFT", "HARD", "MEDIUM"], "tyre_age": [1, 2, 5]}))
    assert out[0] == 2.0 and out[1] == 10.0 and out[2] == pytest.approx(np.median([1, 2, 3, 10, 10, 10]))


@pytest.mark.parametrize("algorithm", ["random_forest", "lightgbm"])
def test_regressors_learn_signal_and_handle_nan(algorithm):
    X, y, _ = make_features()
    params = {"n_estimators": 60, "min_samples_leaf": 3} if algorithm == "random_forest" else {"n_estimators": 100, "objective": "huber"}
    model = build_regressor(algorithm, params, seed=1).fit(X, y)
    assert regression_metrics(y.to_numpy(), model.predict(X))["r2"] > 0.8
    assert np.isfinite(model.predict(X.iloc[[0]])).all()


@pytest.mark.parametrize("algorithm", ["random_forest", "lightgbm"])
def test_classifiers_return_probabilities(algorithm):
    X, _, c = make_features()
    model = build_classifier(algorithm, {"n_estimators": 60}, {"n_estimators": 80}, seed=1).fit(X, c)
    proba = model.predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert classification_metrics(c.to_numpy(), proba, 0.5)["roc_auc"] > 0.9


def test_unknown_algorithm_raises():
    with pytest.raises(ValueError):
        build_regressor("svm", {}, 0)


def test_lgbm_search_is_reproducible_and_distinct():
    a, b = sample_lgbm_params(8, seed=3), sample_lgbm_params(8, seed=3)
    assert a == b and len({tuple(sorted(t.items())) for t in a}) == 8


def test_best_f1_threshold_separates_classes():
    y = np.array([0] * 50 + [1] * 50)
    proba = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    t = best_f1_threshold(y, proba)
    assert 0.4 <= t <= 0.6
    assert classification_metrics(y, proba, t)["f1"] == pytest.approx(1.0)


def test_classification_metrics_single_class_is_nan_not_crash():
    out = classification_metrics(np.zeros(10), np.linspace(0, 1, 10), 0.5)
    assert np.isnan(out["roc_auc"]) and out["prevalence"] == 0.0


def test_bundle_output_schema_and_threshold():
    X, y, c = make_features()
    reg = build_regressor("lightgbm", {"n_estimators": 50}, 1).fit(X, y)
    clf = build_classifier("lightgbm", {}, {"n_estimators": 50}, 1).fit(X, c)
    out = PitStrategyModel(reg, clf, threshold=0.5).predict_frame(X.iloc[:10])
    assert list(out.columns) == ["pace_loss_s", "pit_probability", "pit_recommended"]
    assert out["pit_recommended"].dtype == bool
    assert (out["pit_recommended"] == (out["pit_probability"] >= 0.5)).all()


def test_latency_helper_returns_ordered_percentiles():
    X, _, _ = make_features(50)
    out = single_row_latency_ms(lambda row: row.sum(numeric_only=True), X, 20)
    assert 0 <= out["latency_ms_p50"] <= out["latency_ms_p95"]


def test_group_rate_baseline_returns_pit_rate_per_cell_with_fallback():
    X = pd.DataFrame({"compound": ["SOFT"] * 4 + ["HARD"] * 4, "tyre_age": [5, 5, 5, 5, 9, 9, 9, 9]})
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    model = GroupRateBaseline().fit(X, y)
    proba = model.predict_proba(pd.DataFrame({"compound": ["SOFT", "HARD", "MEDIUM"], "tyre_age": [5, 9, 3]}))
    assert proba.shape == (3, 2) and np.allclose(proba.sum(axis=1), 1.0)
    assert proba[0, 1] == pytest.approx(0.75) and proba[1, 1] == 0.0
    assert proba[2, 1] == pytest.approx(3 / 8)  # unseen cell falls back to the global rate
    assert list(model.predict(pd.DataFrame({"compound": ["SOFT"], "tyre_age": [5]}))) == [1]


def test_lightgbm_results_do_not_depend_on_thread_count():
    X, y, c = make_features(1500, seed=5)
    params = {"n_estimators": 80, "objective": "huber", "num_leaves": 63, "subsample": 0.7, "colsample_bytree": 0.6}

    def fit_predict(n_jobs: int) -> np.ndarray:
        model = build_regressor("lightgbm", params, seed=11)
        model.set_params(model__n_jobs=n_jobs)
        return model.fit(X, y).predict(X)

    assert np.array_equal(fit_predict(1), fit_predict(2))


def test_bundle_rejects_wet_compound_rows():
    X, y, c = make_features(300)
    reg = build_regressor("lightgbm", {"n_estimators": 20}, 1).fit(X, y)
    clf = build_classifier("lightgbm", {}, {"n_estimators": 20}, 1).fit(X, c)
    bundle = PitStrategyModel(reg, clf, threshold=0.5)
    with pytest.raises(ValueError, match="dry races only"):
        bundle.predict_frame(X.iloc[:2].assign(compound="WET"))
