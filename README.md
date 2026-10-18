# F1-PitStrategy

Predicts tyre-wear pace loss and whether a pit stop should be considered, using Formula 1 lap, stint, and weather data from OpenF1.

## Included in this branch

- OpenF1 ingestion, preprocessing, and DVC pipeline.
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
ruff check src tests
```

The API needs `models/champion`. If you have access to the DVC remote, run `dvc pull` first.

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

## Important notes

- The model is trained on dry races only. Rainfall may trigger a warning, but wet tyre compounds are rejected.
- The pit recommendation is a heuristic, not actual team strategy.
- The API has no authentication or rate limiting. Put it behind a gateway before exposing it outside a trusted network.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy or a shared remote to retrieve data and model files.
- The API Docker image and GitHub Actions workflows have not yet been verified on GitHub. The real model also needs to be available from DVC for a full deployment check.
