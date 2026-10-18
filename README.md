# F1-PitStrategy

This project predicts tyre-wear pace loss and a pit recommendation from Formula 1 data.

## What is included

- Downloads lap, stint, weather, race-control, and session data from OpenF1.
- Saves raw data by session and retries failed or incomplete requests.
- Removes unsuitable laps and wet or mixed-weather sessions.
- Builds pace-loss and heuristic pit recommendation targets, plus past-lap features.
- Splits the data by season: 2023 and 2024 for training, 2025 for testing.
- Tracks the pipeline and data with DVC.
- Regression and classification training, evaluation, and MLflow tracking.
- FastAPI prediction service with single-lap, batch, and history-based endpoints.
- Docker and Docker Compose setup.
- GitHub Actions checks for linting, tests, and a Docker smoke test.
- Evidently input-drift reports and a weekly Airflow retraining DAG.

## Setup

Requirements: Python 3.11, Git, DVC, and Docker.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
python -m src.data.ingest
python -m src.data.preprocess
```

Run commands from the repository root. To run the configured pipeline, use `dvc repro`. The API needs `models/champion`; if you have access to the DVC remote, run `dvc pull` first.

## Data notes

- The pipeline is designed for dry races. Wet and mixed-weather sessions are excluded.
- The pit recommendation is a heuristic based on recent pace loss, not actual team strategy.
- Data files are managed by DVC and are not committed directly to Git.

## Run the API

```bash
docker compose up --build -d
```

The API is available at `http://localhost:8000`, with interactive docs at `http://localhost:8000/docs`.

To start the API and Airflow:

```bash
cp .env.example .env
# Set AIRFLOW_UID to the output of: id -u
docker compose --profile airflow up --build -d
```

Airflow is available at `http://localhost:8080`. The weekly DAG checks for input drift and can run ingestion, preprocessing, training, and a summary step. Set the `force_retrain` DAG parameter to skip the drift check and start retraining.

## Monitoring and CI

The drift command compares logged API inputs with the training data:

```bash
python -m src.monitoring.drift
```

Reports are written to `reports/drift/`. Fewer than 200 logged requests is treated as insufficient data. The thresholds are initial values and should be reviewed against real traffic.

GitHub Actions runs linting and tests, a Docker smoke test using a temporary synthetic model, and Airflow DAG tests. The manual image-publish workflow needs a shared DVC remote and repository secrets before it can be used.

## Tests

Run `python -m pytest -q`. Tests use synthetic data and do not need network access.

## Important notes
- The model is trained on dry races only. Rainfall may trigger a warning, but wet tyre compounds are rejected.
- The pit recommendation is a heuristic, not actual team strategy.
- The API has no authentication or rate limiting. Put it behind a gateway before exposing it outside a trusted network.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy or a shared remote to retrieve data and model files.
- The CI Docker smoke test uses a temporary synthetic model. The manual image-publish workflow has not been run, and a full deployment check needs the real model from a shared DVC remote.
