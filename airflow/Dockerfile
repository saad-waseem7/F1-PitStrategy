# Airflow for the retrain DAG. The project runs in its OWN virtualenv so Airflow's pinned packages and the
# project's (mlflow, pandas, evidently) never have to agree.
#   docker compose --profile airflow up --build -d      (from the repository root)
FROM apache/airflow:2.10.5-python3.11
ENV PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10
COPY --chown=airflow:root requirements.txt /tmp/f1-requirements.txt
RUN python -m venv /home/airflow/f1-venv \
    && /home/airflow/f1-venv/bin/pip install -r /tmp/f1-requirements.txt
ENV F1_PYTHON=/home/airflow/f1-venv/bin/python \
    F1_PROJECT_DIR=/opt/f1
