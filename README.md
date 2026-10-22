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

## Web app

With the API running, open `http://localhost:8000/`. Four pages:

- **Predict**: enter a lap, optionally with the stint's earlier lap times, and get pace loss, pit probability and a box or stay out call.
- **Batch**: upload or paste a CSV of laps (the template button gives the columns), score them in one request, and download the calls as a CSV. Row numbers in error messages count laps after the header.
- **Stint replay**: an overview of how the model scores a whole sample (error, bias, R², pit flag precision and recall, per race and per tyre compound), then a per-stint view comparing the model's per-lap output with what actually happened. The sample is built from the held-out test split and keeps every race by default: `python -m src.replay.build_sample` (use `--races N` for a smaller one), then commit `replay/sample.json.gz` and run `docker compose restart api`. Replay calls are not written to the prediction log, so they do not affect drift checks.
- **Model & monitoring**: the champion's version, held-out metrics and the latest drift report. The report appears after the Airflow DAG or `python -m src.monitoring.drift` has run.

The page is plain HTML, CSS and JavaScript in `web/`, with no build step. Fonts (Barlow Semi Condensed and IBM Plex Sans, both SIL Open Font License) are served from `web/assets/fonts/`, so nothing loads from a CDN. It follows the operating system's light or dark setting. `web/`, `replay/` and `reports/` are bind-mounted into the container, so edits to `web/` show up on a browser refresh with no rebuild. Changes to Python code still need `docker compose up --build -d`. It is served with a strict content security policy and never inserts strings as HTML. It has no login, so keep it on a trusted network.

## Tests

Run `python -m pytest -q`. Tests use synthetic data and do not need network access.

## Important notes
- The model is trained on dry races only. Rainfall may trigger a warning, but wet tyre compounds are rejected.
- The pit recommendation is a heuristic, not actual team strategy.
- The API has no authentication or rate limiting. Put it behind a gateway before exposing it outside a trusted network.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy or a shared remote to retrieve data and model files.
- The CI Docker smoke test uses a temporary synthetic model. The manual image-publish workflow has not been run, and a full deployment check needs the real model from a shared DVC remote.
