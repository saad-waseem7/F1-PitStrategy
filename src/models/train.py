"""Train, compare, evaluate, register and export the pit strategy model.

Flow (everything is logged to MLflow):
  1. Group-by-race cross-validation on the training years only: naive baseline vs Random Forest
     vs tuned LightGBM for pace loss (seconds). The lowest out-of-fold MAE wins.
  2. A classifier of the same family predicts the pit flag; its probability cut-off maximises F1
     on out-of-fold predictions (never on the test set).
  3. Final fit on all training rows, one evaluation on the held-out test year.
  4. Gates: the regressor must beat the naive baseline by min_lift_over_baseline, and the
     classifier must beat a compound-and-tyre-age lookup by min_pr_auc_lift_over_baseline
     (PR-AUC), else the run fails and nothing is registered.
  5. The bundle (regressor + classifier + threshold) is registered; it takes the `champion`
     alias only if it beats the current champion's test MAE by min_improvement_for_champion.
  6. The current champion is exported to models/champion (the DVC-tracked deliverable).

Usage:
    python -m src.models.train
"""
from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import inspect
import json
import logging
import shutil
import subprocess
from pathlib import Path
from collections.abc import Callable
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupKFold

from src.config import (
    FEATURE_COLUMNS,
    METRICS_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    load_params,
)
from src.features import NUMERIC_COLUMNS
from src.models.bundle import PitStrategyModel
from src.models.estimators import (
    GroupMedianBaseline,
    GroupRateBaseline,
    build_classifier,
    build_regressor,
    sample_lgbm_params,
)
from src.models.evaluate import (
    best_f1_threshold,
    classification_metrics,
    regression_metrics,
    single_row_latency_ms,
)

LOG = logging.getLogger(__name__)
CHAMPION_ALIAS = "champion"


# --------------------------------------------------------------------------- helpers
def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, cwd=PROJECT_ROOT)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def _finite(metrics: dict[str, float]) -> dict[str, float]:
    return {k: v for k, v in metrics.items() if np.isfinite(v)}


def oof_predict(build: Callable[[], Any], X: pd.DataFrame, y: pd.Series, groups: pd.Series, folds: int, proba: bool = False) -> np.ndarray:
    """Out-of-fold predictions with GroupKFold: every race sits entirely inside one fold."""
    out = np.zeros(len(X))
    for train_idx, val_idx in GroupKFold(n_splits=folds).split(X, y, groups):
        model = build()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        out[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1] if proba else model.predict(X.iloc[val_idx])
    return out


def _cv_candidate(name: str, build: Callable[[], Any], params: dict[str, Any], X, y, groups, folds: int) -> float:
    with mlflow.start_run(run_name=name, nested=True):
        mlflow.log_params({k: str(v) for k, v in params.items()})
        metrics = regression_metrics(y.to_numpy(), oof_predict(build, X, y, groups, folds))
        mlflow.log_metrics({f"cv_{k}": v for k, v in metrics.items()})
    LOG.info("%-18s CV MAE %.4f", name, metrics["mae"])
    return metrics["mae"]


def select_regressor(X, y, groups, tp: dict[str, Any]) -> dict[str, Any]:
    seed, folds = tp["seed"], tp["cv_folds"]
    baseline_mae = _cv_candidate("baseline_median", GroupMedianBaseline, {}, X, y, groups, folds)
    rf_params = dict(tp["rf"])
    rf_mae = _cv_candidate("random_forest", lambda: build_regressor("random_forest", rf_params, seed), rf_params, X, y, groups, folds)

    best_lgbm: tuple[float, dict[str, Any]] = (float("inf"), {})
    for i, trial in enumerate(sample_lgbm_params(tp["lgbm_trials"], seed)):
        params = {"objective": tp["lgbm_objective"], **trial}
        mae = _cv_candidate(f"lightgbm_trial_{i:02d}", lambda p=params: build_regressor("lightgbm", p, seed), params, X, y, groups, folds)
        if mae < best_lgbm[0]:
            best_lgbm = (mae, params)

    algorithm = "lightgbm" if best_lgbm[0] <= rf_mae else "random_forest"
    return {
        "algorithm": algorithm,
        "params": best_lgbm[1] if algorithm == "lightgbm" else rf_params,
        "cv_mae": {"baseline": baseline_mae, "random_forest": rf_mae, "lightgbm": best_lgbm[0]},
        "lgbm_params": best_lgbm[1],
    }


def _model_input_example(X: pd.DataFrame) -> pd.DataFrame:
    sample = X.head(5).copy()
    sample["compound"] = sample["compound"].astype(str)
    sample[list(NUMERIC_COLUMNS)] = sample[list(NUMERIC_COLUMNS)].astype("float64")
    return sample


def _log_bundle(bundle: PitStrategyModel, X_sample: pd.DataFrame, model_name: str) -> Any:
    sample = _model_input_example(X_sample)
    kwargs: dict[str, Any] = dict(
        python_model=bundle,
        # No inferred signature: MLflow's would reject integer columns (JSON clients send ints) and
        # would mark columns required or optional from five sample rows. FeaturePreparer is the
        # contract: it casts types and raises a clear ValueError for missing columns or wet tyres.
        signature=False,
        input_example=sample,
        code_paths=[str(PROJECT_ROOT / "src")],  # ship src/ so the model loads without the repo installed
        pip_requirements=[f"{p}=={metadata.version(p)}" for p in ("mlflow", "scikit-learn", "lightgbm", "pandas", "numpy")],
        registered_model_name=model_name,
    )
    # MLflow 3 renamed artifact_path to name.
    kwargs["name" if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters else "artifact_path"] = "model"
    return mlflow.pyfunc.log_model(**kwargs)


def _registered_version(client: MlflowClient, info: Any, model_name: str) -> str:
    version = getattr(info, "registered_model_version", None)
    if version:
        return str(version)
    return str(max(int(v.version) for v in client.search_model_versions(f"name='{model_name}'")))


def maybe_promote(
    client: MlflowClient,
    model_name: str,
    version: str,
    test_mae: float,
    min_improvement: float,
    data_md5: str,
) -> bool:
    """Give the `champion` alias to `version` when that is safe.

    Promote if there is no champion, if the candidate is clearly better (min_improvement), or if it was trained
    and tested on the same data as the champion and is not worse. The last case lets a code or parameter change
    on unchanged data replace the old model, so the exported champion always comes from the current pipeline.
    A candidate that is worse on identical data, or only marginally better on different data, is not promoted.
    """
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        champion = None
    if champion is None or "test_mae" not in champion.tags:
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
        return True
    champion_mae = float(champion.tags["test_mae"])
    run_tags = client.get_run(champion.run_id).data.tags
    same_data = f"{run_tags.get('train_data_md5')}:{run_tags.get('test_data_md5')}" == data_md5
    test_mae = round(test_mae, 6)  # the champion's tag is stored with 6 decimals; compare like with like
    if test_mae <= champion_mae * (1.0 - min_improvement) or (same_data and test_mae <= champion_mae):
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version)
        return True
    LOG.info(
        "Not promoted: test MAE %.4f vs champion %.4f (same data: %s, needs %.0f%% better otherwise)",
        test_mae, champion_mae, same_data, min_improvement * 100,
    )
    return False


def export_champion(model_name: str, models_dir: Path) -> Path:
    dst = models_dir / "champion"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{model_name}@{CHAMPION_ALIAS}", dst_path=str(dst))
    return dst


def _setup_mlflow(cfg: dict[str, Any]) -> None:
    mlflow.set_tracking_uri(cfg["tracking_uri"])
    if mlflow.get_experiment_by_name(cfg["experiment"]) is None:
        root = (PROJECT_ROOT / cfg["artifact_root"]).resolve()
        mlflow.create_experiment(cfg["experiment"], artifact_location=root.as_uri())
    mlflow.set_experiment(cfg["experiment"])


# --------------------------------------------------------------------------- main
def run(
    data_dir: Path = PROCESSED_DIR,
    metrics_dir: Path = METRICS_DIR,
    models_dir: Path = MODELS_DIR,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    params = params or load_params()
    tp = params["train"]
    seed = int(tp["seed"])
    np.random.seed(seed)

    train = pd.read_parquet(data_dir / "train.parquet")
    test = pd.read_parquet(data_dir / "test.parquet")
    cols = list(FEATURE_COLUMNS)
    X_train, X_test = train[cols], test[cols]
    y_train, y_test = train[TARGET_REGRESSION], test[TARGET_REGRESSION]
    c_train, c_test = train[TARGET_CLASSIFICATION].astype(int), test[TARGET_CLASSIFICATION].astype(int)
    groups = train["session_key"]
    if groups.nunique() < tp["cv_folds"]:
        raise ValueError(f"Need at least {tp['cv_folds']} race sessions for CV, found {groups.nunique()}")

    _setup_mlflow(tp["mlflow"])
    model_name = tp["mlflow"]["model_name"]
    client = MlflowClient()

    train_md5, test_md5 = _md5(data_dir / "train.parquet"), _md5(data_dir / "test.parquet")
    with mlflow.start_run(run_name=f"train-{_git_commit()}") as parent:
        mlflow.log_params(_flatten(tp))
        mlflow.set_tags(
            {
                "git_commit": _git_commit(),
                "train_data_md5": train_md5,
                "test_data_md5": test_md5,
                "train_rows": str(len(train)),
                "test_rows": str(len(test)),
            }
        )

        # 1. Regressor selection by grouped CV
        choice = select_regressor(X_train, y_train, groups, tp)
        algorithm = choice["algorithm"]
        LOG.info("Selected %s (CV MAE %s)", algorithm, {k: round(v, 4) for k, v in choice["cv_mae"].items()})

        # 2. Classifier of the same family, threshold from out-of-fold probabilities
        def make_classifier() -> Any:
            return build_classifier(algorithm, tp["rf"], tp["lgbm_classifier"], seed)

        oof_proba = oof_predict(make_classifier, X_train, c_train, groups, tp["cv_folds"], proba=True)
        threshold = best_f1_threshold(c_train.to_numpy(), oof_proba)
        cv_cls = classification_metrics(c_train.to_numpy(), oof_proba, threshold)

        # 3. Final fit and a single test evaluation
        regressor = build_regressor(algorithm, choice["params"], seed).fit(X_train, y_train)
        classifier = make_classifier().fit(X_train, c_train)
        baseline = GroupMedianBaseline().fit(X_train, y_train)
        test_reg = regression_metrics(y_test.to_numpy(), regressor.predict(X_test))
        baseline_reg = regression_metrics(y_test.to_numpy(), baseline.predict(X_test))
        test_cls = classification_metrics(c_test.to_numpy(), classifier.predict_proba(X_test)[:, 1], threshold)
        lift = (baseline_reg["mae"] - test_reg["mae"]) / baseline_reg["mae"]

        # Classifier baseline: pit rate per (compound, tyre age), same protocol as the model
        # (threshold from out-of-fold predictions on the training years only).
        base_oof = oof_predict(GroupRateBaseline, X_train, c_train, groups, tp["cv_folds"], proba=True)
        base_threshold = best_f1_threshold(c_train.to_numpy(), base_oof)
        base_cls = classification_metrics(
            c_test.to_numpy(), GroupRateBaseline().fit(X_train, c_train).predict_proba(X_test)[:, 1], base_threshold
        )
        pr_auc_lift = test_cls["pr_auc"] - base_cls["pr_auc"]

        bundle = PitStrategyModel(regressor, classifier, threshold)
        latency = single_row_latency_ms(bundle.predict_frame, X_test, int(tp["latency_samples"]))

        sample = X_test.sample(min(int(tp["permutation_sample"]), len(X_test)), random_state=seed)
        sample = sample.assign(compound=sample["compound"].astype(str))
        importance = permutation_importance(
            regressor, sample, y_test.loc[sample.index], scoring="neg_mean_absolute_error", n_repeats=3, random_state=seed, n_jobs=1
        )
        ranking = sorted(zip(cols, importance.importances_mean, strict=True), key=lambda kv: kv[1], reverse=True)
        top_features = [{"feature": f, "mae_increase_s": round(float(v), 5)} for f, v in ranking]

        mlflow.log_metrics(_finite({f"cv_cls_{k}": v for k, v in cv_cls.items()}))
        mlflow.log_metrics(_finite({f"test_{k}": v for k, v in test_reg.items()}))
        mlflow.log_metrics(_finite({f"test_cls_{k}": v for k, v in test_cls.items()}))
        mlflow.log_metrics(_finite({f"baseline_test_cls_{k}": v for k, v in base_cls.items()}))
        mlflow.log_metrics(
            _finite({"baseline_test_mae": baseline_reg["mae"], "lift_over_baseline": lift, "pr_auc_lift_over_baseline": pr_auc_lift, **latency})
        )
        mlflow.log_dict({"permutation_importance": top_features}, "permutation_importance.json")
        mlflow.set_tag("selected_algorithm", algorithm)

        report: dict[str, Any] = {
            "selected_algorithm": algorithm,
            "cv_mae": choice["cv_mae"],
            "test_regression": test_reg,
            "baseline_test_mae": baseline_reg["mae"],
            "lift_over_baseline": lift,
            "cv_classification": cv_cls,
            "test_classification": test_cls,
            "baseline_test_classification": base_cls,
            "pr_auc_lift_over_baseline": pr_auc_lift,
            "latency": latency,
            "top_features": top_features[:5],
            "mlflow_run_id": parent.info.run_id,
            "promoted": False,
        }

        # 4. Gates: refuse to register a model that does not clearly beat the naive baselines
        failures = []
        if lift < tp["min_lift_over_baseline"]:
            failures.append(
                f"test MAE {test_reg['mae']:.4f} is only {lift:.1%} better than the naive baseline "
                f"({baseline_reg['mae']:.4f}); required {tp['min_lift_over_baseline']:.0%}"
            )
        if np.isfinite(pr_auc_lift) and pr_auc_lift < tp["min_pr_auc_lift_over_baseline"]:
            failures.append(
                f"test PR-AUC {test_cls['pr_auc']:.3f} is only {pr_auc_lift:+.3f} above the compound-and-tyre-age "
                f"baseline ({base_cls['pr_auc']:.3f}); required {tp['min_pr_auc_lift_over_baseline']:+.3f}"
            )
        if failures:
            _write_report(report, metrics_dir)
            raise RuntimeError("Model rejected: " + "; ".join(failures) + ".")

        # 5. Register, tag, maybe promote
        info = _log_bundle(bundle, X_train, model_name)
        version = _registered_version(client, info, model_name)
        client.set_model_version_tag(model_name, version, "test_mae", f"{test_reg['mae']:.6f}")
        client.set_model_version_tag(model_name, version, "algorithm", algorithm)
        report["promoted"] = maybe_promote(
            client, model_name, version, test_reg["mae"], tp["min_improvement_for_champion"], f"{train_md5}:{test_md5}"
        )
        report["model_version"] = version
        report["champion_version"] = str(client.get_model_version_by_alias(model_name, CHAMPION_ALIAS).version)
        mlflow.set_tag("promoted_to_champion", str(report["promoted"]))

    # 6. Export the current champion for DVC and the serving container
    export_champion(model_name, models_dir)
    _write_report(report, metrics_dir)
    LOG.info(
        "Test MAE %.4f s (baseline %.4f, lift %.1f%%), PR-AUC %.3f (baseline %.3f), p95 latency %.1f ms, promoted=%s",
        test_reg["mae"], baseline_reg["mae"], lift * 100, test_cls["pr_auc"], base_cls["pr_auc"],
        latency["latency_ms_p95"], report["promoted"],
    )
    return report


def _write_report(report: dict[str, Any], metrics_dir: Path) -> None:
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "train_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
