"""Weekly check for drift in served requests, and retraining when it is found.

    choose_path --force_retrain--------------------------------------------+
        |                                                                   |
        +--> check_drift --> decide_after_drift --drift--> ingest --> preprocess --> train --> summarize
                                       |
                                       +--no drift--> skip_retrain

Every task shells out to the project's own virtualenv (F1_PYTHON) in the project directory (F1_PROJECT_DIR),
so Airflow's dependency pins never meet the project's (mlflow, pandas, evidently). Each command is the same
one `dvc repro` runs, so the DAG and a manual run behave identically.

The DAG writes data/raw, data/processed, models/champion and metrics/ in the project directory. It does NOT
commit anything: a person reviews the result, then runs `dvc status`, commits `dvc.lock` and `metrics/`, and
pushes. A gate failure in `train` ("Model rejected") fails the DAG on purpose and leaves the old champion.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import timedelta
from pathlib import Path

import pendulum
from airflow.exceptions import AirflowFailException
from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator

LOG = logging.getLogger(__name__)

PROJECT_DIR = os.environ.get("F1_PROJECT_DIR", "/opt/f1")
PYTHON = os.environ.get("F1_PYTHON", "/opt/f1-venv/bin/python")
DRIFT_SUMMARY = Path(PROJECT_DIR) / "reports" / "drift" / "drift_summary.json"
TRAIN_REPORT = Path(PROJECT_DIR) / "metrics" / "train_metrics.json"


def choose_path(params: dict) -> str:
    return "ingest" if params.get("force_retrain") else "check_drift"


def after_drift(summary_path: Path = DRIFT_SUMMARY) -> str:
    """Branch on the drift summary. A missing or unreadable summary fails the run instead of guessing."""
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        detected = bool(summary["drift_detected"])
    except (OSError, ValueError, KeyError) as exc:
        raise AirflowFailException(f"No usable drift summary at {summary_path}: {exc}") from exc
    LOG.info(
        "drift status=%s detected=%s columns=%s", summary.get("status"), detected, summary.get("drifted_columns")
    )
    return "ingest" if detected else "skip_retrain"


def summarize(report_path: Path = TRAIN_REPORT) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    result = {k: report.get(k) for k in ("selected_algorithm", "model_version", "champion_version", "promoted")}
    result["test_mae"] = (report.get("test_regression") or {}).get("mae")
    result["test_pr_auc"] = (report.get("test_classification") or {}).get("pr_auc")
    LOG.info("Retrain finished: %s", result)
    return result


def _cmd(command: str) -> str:
    return f'cd "{PROJECT_DIR}" && {command}'


with DAG(
    dag_id="f1_retrain",
    description="Drift check on served requests; retrain through the same gates as a manual run",
    schedule="@weekly",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    params={"force_retrain": Param(False, type="boolean", description="Skip the drift check and retrain now")},
    default_args={"owner": "mlops", "retries": 0},
    tags=["f1", "mlops"],
) as dag:
    start = BranchPythonOperator(
        task_id="choose_path",
        python_callable=lambda params, **_: choose_path(params),
    )
    check_drift = BashOperator(
        task_id="check_drift",
        # Remove the previous summary first so a failed check can never be mistaken for a fresh answer.
        bash_command=_cmd(
            f'rm -f "{DRIFT_SUMMARY}" && "{PYTHON}" -m src.monitoring.drift --drift-share "${{F1_DRIFT_SHARE:-0.2}}"'
        ),
    )
    decide = BranchPythonOperator(task_id="decide_after_drift", python_callable=lambda **_: after_drift())
    skip = EmptyOperator(task_id="skip_retrain")
    # Throttled API calls and network drops are expected during ingest; training failures are not retried.
    ingest = BashOperator(
        task_id="ingest",
        bash_command=_cmd(f'"{PYTHON}" -m src.data.ingest'),
        trigger_rule="none_failed_min_one_success",
        retries=2,
        retry_delay=timedelta(minutes=5),
    )
    preprocess = BashOperator(task_id="preprocess", bash_command=_cmd(f'"{PYTHON}" -m src.data.preprocess'))
    train = BashOperator(task_id="train", bash_command=_cmd(f'"{PYTHON}" -m src.models.train'))
    report = PythonOperator(task_id="summarize", python_callable=lambda **_: summarize())

    start >> [check_drift, ingest]
    check_drift >> decide >> [ingest, skip]
    ingest >> preprocess >> train >> report
