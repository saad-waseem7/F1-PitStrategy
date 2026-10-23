# F1-PitStrategy

A complete MLOps project (CS4085, FAST-NUCES Lahore) that predicts how much pace a Formula 1 car loses to tyre wear on a given lap, and whether to pit. It covers data ingestion, a versioned pipeline, model training with tracking and a registry, a REST API, a web app, drift monitoring and automated retraining.

The pit recommendation is a heuristic, not real team strategy, and the model only supports dry races. See [Limitations](#limitations) before relying on any number here.

## Results

Trained on 2023 and 2024 races, tested on 2025 (21 races, 18,373 laps). Model selection used cross-validation grouped by race, so the 2025 figures are report-only.

| | Naive baseline | Random forest | LightGBM (champion, v5) |
|---|---|---|---|
| Pace loss, CV MAE (s) | 0.492 | 0.381 | **0.368** |
| Pace loss, test MAE (s) | 0.531 | | **0.375** |
| Pace loss, test R² | | | 0.487 |
| Pit flag, test PR-AUC | 0.448 | | **0.884** |
| Pit flag, test F1 | 0.524 | | 0.813 |

The baseline looks up the median (or pit rate) for each tyre compound and tyre age. The champion cuts test MAE by 29% against it. Single-lap prediction latency is about 11 ms (p50) and 13 ms (p95) on CPU.

One result is not flattering: a persistence baseline (score a lap by the actual pace loss of the previous laps in the stint) reaches test PR-AUC 0.951, above the model's 0.884. That baseline reads true lap times against the stint's best laps, which a live service cannot know, and the pit label is itself a trailing mean of pace loss, so the baseline partly reads the answer. It is reported, not gated.

## What is included

- Ingestion of laps, stints, weather, race-control and session data from OpenF1, with retries and per-session raw files.
- Preprocessing that removes unsuitable laps and wet or mixed sessions, then builds the pace-loss target, the heuristic pit label and lag features.
- A DVC pipeline (`ingest`, `preprocess`, `train`) with metrics tracked in Git.
- Training of a naive baseline, a random forest and LightGBM with MLflow tracking, a model registry and quality gates before a champion is promoted.
- A FastAPI service: single-lap, batch and history-based endpoints, request IDs, strict validation and a prediction log.
- The Pit Wall web app: Predict, Batch, Stint Replay, Model and Monitoring.
- Docker and Docker Compose, with an optional Airflow service.
- Evidently drift reports and a weekly Airflow DAG that retrains when drift is found.
- GitHub Actions for linting, tests, a Docker smoke test and the Airflow DAG tests.

## Quick start

Requirements: Python 3.11, Git, Docker with the Compose plugin, and DVC.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The tests use synthetic data and need no network. To run the app you also need the trained model in `models/champion`. It is not stored in Git. See [Getting the data and model](#getting-the-data-and-model).

```bash
docker compose up --build -d      # builds the image and starts the API
```

Open `http://localhost:8000/` for the web app and `http://localhost:8000/docs` for the interactive API docs. Stop with `docker compose down`.

## Getting the data and model

Data, the processed tables and `models/champion` are tracked by DVC and are not in Git. The DVC remote configured in `.dvc/config` is a folder on the project owner's machine (`/home/saad/Documents/dvc-storage`), so `dvc pull` does not work for anyone else. There are two ways to get the files:

1. **Rebuild everything.** Run `dvc repro`. This calls OpenF1 for every 2023 to 2025 race, waits 2.5 s between requests and retries throttled calls, so ingestion is slow. Training is seeded (`seed: 42`), but OpenF1 can change its data, so the numbers may differ slightly from the table above.
2. **Use your own remote.** Point DVC at storage you control, then push and pull:

   ```bash
   dvc remote modify localstore url /path/to/your/storage    # or: dvc remote add -d myremote <url>
   dvc push        # on the machine that has the data
   dvc pull        # on the machine that needs it
   ```

CI never needs either: its Docker smoke test builds a small synthetic champion with `tests/make_ci_champion.py`.

## Pipeline

`dvc repro` runs three stages from `dvc.yaml`. Every setting lives in `params.yaml`.

| Stage | Command | Output |
|---|---|---|
| `ingest` | `python -m src.data.ingest` | `data/raw/` |
| `preprocess` | `python -m src.data.preprocess` | `data/processed/`, `metrics/preprocess_report.json` |
| `train` | `python -m src.models.train` | `models/champion/`, `metrics/train_metrics.json` |

**Data.** 77,075 raw laps become 51,833 modelled laps across 61 race sessions (33,460 train, 18,373 test). Filters remove missing values, 9 wet or mixed sessions, lap 1, pit in and out laps, safety-car laps and laps slower than 107% of the driver's median.

**Target.** Pace loss is a lap's fuel-corrected time minus the stint's reference pace (the 10th percentile of clean laps in that stint). The pit flag is set when the trailing 3-lap mean pace loss is at least 1.0 s. The label is a heuristic. About 26% of laps carry it.

**Features (15).** Tyre age, compound, stint number, lap number, race progress, track temperature, air temperature, humidity, rainfall, wind speed and the change in track temperature over 5 laps, plus four lag features that use only laps before the current one in the same stint: the change between the last two laps, the trailing pace-decay slope over 5 laps, the previous lap's gap to the best lap seen so far in the stint, and the mean of that gap over the last 2 laps. The lag features are missing early in a stint. The API computes them from earlier lap times (`/predict/from-history`) or accepts them directly.

**Training and gates.** Candidates are scored with 5-fold cross-validation grouped by race session, so no race appears in both train and validation. The run fails on purpose when:

- cross-validated MAE is not at least 5% better than the baseline, or
- cross-validated PR-AUC is not above the compound and tyre-age lookup.

A new model replaces the champion only when its CV MAE is at least 1% better on new data (on identical data, not worse is enough). Runs are logged to MLflow (`sqlite:///mlflow.db`, experiment `f1-pit-strategy`). The champion is exported to `models/champion` with its own `requirements.txt`, so the serving image uses the same library versions that trained it.

## API

| Method and path | Purpose |
|---|---|
| `GET /health` | Process is up |
| `GET /ready` | Model is loaded (503 otherwise) |
| `GET /model-info` | Version, algorithm, threshold, features, held-out metrics |
| `POST /predict` | One lap; lag features optional |
| `POST /predict/from-history` | One lap plus earlier clean laps of the stint; the service computes the lag features |
| `POST /predict/batch` | Up to 500 laps (`MAX_BATCH`); larger requests get 413 |

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "tyre_age": 14, "compound": "MEDIUM", "stint_number": 1, "lap_number": 18,
  "race_progress": 0.316, "track_temperature": 38, "air_temperature": 27,
  "humidity": 45, "wind_speed": 2.5
}'
```

The response holds `pace_loss_s`, `pit_probability`, `pit_recommended`, `warnings`, the decision `threshold` and a `request_id` (also sent back in the `X-Request-ID` header). Inputs are validated strictly: values out of range, unknown fields and wet compounds are rejected with a 422 that names the field. Rainfall above 0 is accepted but returns a warning.

Each prediction is appended to a JSONL log (`logs/predictions.jsonl`). The drift check reads it.

## Web app

With the API running, open `http://localhost:8000/`. Four pages:

- **Predict**: enter a lap, optionally with the stint's earlier lap times, and get pace loss, pit probability and a box or stay out call.
- **Batch**: upload or paste a CSV of laps (the template button gives the columns), score them in one request and download the calls as a CSV. Row numbers in error messages count laps after the header.
- **Stint Replay**: an overview of how the model scores a whole sample (error, bias, R², pit flag precision and recall, per race and per tyre compound), then a per-stint view comparing the model's per-lap output with what actually happened. Replay calls are not written to the prediction log, so they do not affect drift checks.
- **Model & Monitoring**: the champion's version, held-out metrics and the latest drift report.

The replay sample is built from the held-out 2025 test split and keeps every race: `python -m src.replay.build_sample` (use `--races N` for fewer), then commit `replay/sample.json.gz` and run `docker compose restart api`.

The page is plain HTML, CSS and JavaScript in `web/`, with no build step. It follows the operating system's light or dark setting. Fonts (Barlow Semi Condensed and IBM Plex Sans, SIL Open Font License) are served from `web/assets/fonts/`, so nothing loads from a CDN. It is served with a strict content security policy and never inserts strings as HTML.

`web/`, `replay/` and `reports/` are bind-mounted into the container. Edits to `web/` show up on a browser refresh with no rebuild. Python changes still need `docker compose up --build -d`.

## Monitoring and retraining

```bash
python -m src.monitoring.drift
```

compares logged API inputs with the training data and writes reports to `reports/drift/`. Numeric columns use a normalised Wasserstein distance (limit 0.2) and categorical columns a Jensen-Shannon distance (limit 0.1). Drift is declared when at least 20% of monitored columns drift. Fewer than 200 logged requests counts as insufficient data, which is a valid outcome and not an error. Useful flags: `--drift-share`, `--num-threshold`, `--cat-threshold`, `--min-rows`, `--fail-on-drift`. The limits are initial values chosen on synthetic data and should be reviewed against real traffic.

The weekly Airflow DAG `f1_retrain` automates this:

```
choose_path -> check_drift -> decide_after_drift -> ingest -> preprocess -> train -> summarize
                                       \-> skip_retrain      (no drift)
```

Set the `force_retrain` parameter to skip the drift check. Each task runs the same command `dvc repro` runs, so a manual and an automated run behave the same. A model rejected by the training gates fails the DAG on purpose and leaves the old champion in place. The DAG commits nothing: a person reviews the result, runs `dvc status`, commits `dvc.lock` and `metrics/`, and pushes.

```bash
cp .env.example .env              # set AIRFLOW_UID to the output of: id -u
docker compose --profile airflow up --build -d
docker compose exec airflow cat /opt/airflow/standalone_admin_password.txt
```

Airflow is at `http://localhost:8080`, user `admin`, with the generated password above. It runs as a single container with SQLite and the sequential executor, which is enough for a weekly job and not a production setup. The Airflow image is over 2 GB, so it only starts with the `airflow` profile.

## Docker

One `docker-compose.yml` serves both cases. The `api` service always starts. The `airflow` service starts only with `--profile airflow` (or `COMPOSE_PROFILES=airflow` in `.env`). The API logs go to the `api-logs` volume, which Airflow mounts read-only for the drift check.

The API image builds from `models/champion`, so run `dvc pull` (or `dvc repro`) first. It runs as a non-root user and has a health check on `/ready`. `.github/workflows/publish-image.yml` can publish the image manually. It needs a shared DVC remote and repository secrets and has never been run.

## Project layout

```
src/
  api/          FastAPI app, schemas, model service, web routes, prediction log
  data/         ingest and preprocess
  features.py   one shared feature preparer, so training and serving see identical inputs
  models/       estimators, evaluation, training, model export
  monitoring/   Evidently drift check
  replay/       replay sample builder, store and summary metrics
web/            the web app (HTML, CSS, JavaScript, fonts)
replay/         committed replay sample
airflow/        DAG, Airflow image, DAG tests
tests/          unit and API tests, CI synthetic champion, JavaScript probe
params.yaml     every pipeline setting
dvc.yaml        pipeline definition (dvc.lock pins the last run)
metrics/        preprocess and training reports, tracked in Git
```

## Configuration

The API reads environment variables, so the same image runs anywhere.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_DIR` | `models/champion` | Exported champion |
| `METRICS_PATH` | `metrics/train_metrics.json` | Held-out metrics shown by `/model-info` |
| `PREDICTION_LOG` | `logs/predictions.jsonl` | Request log; empty disables logging |
| `MAX_BATCH` | `500` | Largest batch request |
| `WEB_DIR` | `web` | Web app files |
| `REPLAY_PATH` | `replay/sample.json.gz` | Replay sample; empty disables the replay page |
| `DRIFT_SUMMARY_PATH` | `reports/drift/drift_summary.json` | Latest drift report shown in the app |

## Tests and CI

`python -m pytest -q` runs 154 tests on synthetic data with no network. `ruff check src tests airflow` is the lint step.

GitHub Actions runs three jobs on every pull request and on pushes to `main`: lint and tests, a Docker build with smoke tests against a temporary synthetic champion (`/ready`, `/predict`, a wet-compound rejection, and the web page), and the Airflow DAG tests.

## Limitations

- **Dry races only.** Wet and mixed sessions are excluded from training. Wet compounds are rejected, and rainfall above 0 returns a warning that the estimate is unreliable.
- **The pit flag is a heuristic.** It is a threshold on recent pace loss, not a real team decision. Precision and recall measure agreement with that rule only.
- **Moderate pace-loss fit.** Test R² is 0.49. Most of the signal comes from the lag features, so predictions without earlier laps are noticeably weaker. The persistence baseline outperforms the classifier on PR-AUC, as noted under [Results](#results).
- **No authentication or rate limiting.** Put the API behind a gateway before exposing it beyond a trusted network.
- **Drift limits are untuned.** They were set on synthetic data.
- **Not reproducible from a clone alone.** Data and the model come from DVC, whose remote is a local folder. See [Getting the data and model](#getting-the-data-and-model).
- **Publishing and Airflow scale.** The image-publish workflow has never run, and Airflow is a single-container demo setup.
