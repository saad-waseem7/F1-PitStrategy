"""Airflow DAG: it must parse, have the intended shape, and branch correctly. Skipped without Airflow installed
(Airflow cannot share an environment with the project, so it lives outside tests/ and CI runs it in its own job)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("airflow")

from airflow.exceptions import AirflowFailException  # noqa: E402
from airflow.models import DagBag  # noqa: E402

DAG_DIR = Path(__file__).resolve().parents[1] / "dags"


@pytest.fixture(scope="module")
def dag_module():
    spec = importlib.util.spec_from_file_location("f1_retrain", DAG_DIR / "f1_retrain.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def dag():
    bag = DagBag(dag_folder=str(DAG_DIR), include_examples=False)
    assert bag.import_errors == {}
    return bag.dags["f1_retrain"]  # not get_dag(): that needs an initialised Airflow database


def test_dag_shape(dag):
    assert dag.catchup is False and dag.max_active_runs == 1
    edges = {(t.task_id, d) for t in dag.tasks for d in t.downstream_task_ids}
    assert edges == {
        ("choose_path", "check_drift"), ("choose_path", "ingest"),
        ("check_drift", "decide_after_drift"),
        ("decide_after_drift", "ingest"), ("decide_after_drift", "skip_retrain"),
        ("ingest", "preprocess"), ("preprocess", "train"), ("train", "summarize"),
    }


def test_retry_policy_and_commands(dag):
    assert dag.get_task("ingest").retries == 2
    assert dag.get_task("train").retries == 0  # a failed quality gate must not be retried into passing
    assert dag.get_task("ingest").trigger_rule == "none_failed_min_one_success"
    assert "-m src.data.ingest" in dag.get_task("ingest").bash_command
    assert "-m src.data.preprocess" in dag.get_task("preprocess").bash_command
    assert "-m src.models.train" in dag.get_task("train").bash_command
    assert "rm -f" in dag.get_task("check_drift").bash_command  # stale summaries must not survive


def test_choose_path(dag_module):
    assert dag_module.choose_path({"force_retrain": True}) == "ingest"
    assert dag_module.choose_path({"force_retrain": False}) == "check_drift"
    assert dag_module.choose_path({}) == "check_drift"


@pytest.mark.parametrize("detected, expected", [(True, "ingest"), (False, "skip_retrain")])
def test_branch_follows_drift_summary(dag_module, tmp_path, detected, expected):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"drift_detected": detected, "status": "ok", "drifted_columns": []}))
    assert dag_module.after_drift(path) == expected


@pytest.mark.parametrize("content", [None, "not json", "{}", '{"drift_detected": null}'])
def test_missing_or_broken_summary_fails_instead_of_guessing(dag_module, tmp_path, content):
    path = tmp_path / "s.json"
    if content is not None:
        path.write_text(content)
    if content == '{"drift_detected": null}':
        # null is falsy: treated as "no drift", which is what a writer that only knows "no" would emit
        assert dag_module.after_drift(path) == "skip_retrain"
    else:
        with pytest.raises(AirflowFailException):
            dag_module.after_drift(path)


def test_summarize_reads_the_training_report(dag_module, tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({
        "selected_algorithm": "lightgbm", "model_version": "6", "champion_version": "6", "promoted": True,
        "test_regression": {"mae": 0.37}, "test_classification": {"pr_auc": 0.88},
    }))
    out = dag_module.summarize(path)
    assert out["champion_version"] == "6" and out["test_mae"] == 0.37 and out["test_pr_auc"] == 0.88
