# F1-PitStrategy

Predicts how much lap time a Formula 1 car is losing to tyre wear (seconds) and whether the team should consider pitting (true/false), from OpenF1 lap, stint and weather data. Built for CS4085 MLOps (FAST-NUCES Lahore, Fall 2026).

This repository is built in phases. This README describes what is implemented on `main`; sections for later phases are added by the people who own them.

| Phase | Owner | Content |
|---|---|---|
| 1 | Saad | Data ingestion, preprocessing, DVC pipeline |
| 2 | Saad | Model training, MLflow tracking and registry |
| 3 | Bilal | FastAPI service, Docker, Docker Compose |
| 4 | Bilal | GitHub Actions CI/CD, Evidently drift checks, Airflow DAGs |

## Quick start

Requirements: Ubuntu or similar, Python 3.11, git.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q          # unit tests, synthetic data, no network
ruff check src tests         # lint

dvc pull                     # get data from the DVC remote (see "Sharing data" below)
dvc repro                    # run the pipeline; unchanged stages are skipped
```

Always run commands from the repository root. The code imports `src.*`, so running scripts from inside `src/` fails.

## Pipeline

```
OpenF1 API --ingest--> data/raw/<endpoint>/<session>.parquet
                          --preprocess--> data/processed/{features,train,test}.parquet
                                          metrics/preprocess_report.json
                                             --train--> models/champion        (deployable model)
                                                        metrics/train_metrics.json
                                                        mlflow.db, mlartifacts/ (local registry)
```

Stages are defined in `dvc.yaml`; all tunable values live in `params.yaml`.

| Stage | Command | Output |
|---|---|---|
| `ingest` | `python -m src.data.ingest` | `data/raw/` (laps, stints, weather, race_control, sessions) |
| `preprocess` | `python -m src.data.preprocess` | `data/processed/`, `metrics/preprocess_report.json` |
| `train` | `python -m src.models.train` | `models/champion`, `metrics/train_metrics.json` |

Ingestion is resumable: a session whose `laps` file exists is skipped, files are written atomically, and `persist: true` in `dvc.yaml` stops DVC from deleting `data/raw` before a rerun. Throttled, failed, truncated or empty responses are retried after a fixed 3 seconds (a longer `Retry-After` from the server is honoured). After the retries are used up the run fails loudly instead of producing a partial dataset.

A session for which OpenF1 returns no laps, stints and weather even after the retries is skipped with a warning (observed for session 9086). It is retried on every ingest run, which costs about 40 seconds each time, so it is safe to ignore.

## Data definition

**Scope.** Dry races only. A session is dropped if 5% or more of its laps are on wet tyres or in rain, and any wet-tyre lap left in a dry session is removed. The model is not valid for wet or mixed conditions.

**Laps removed.** Lap 1, pit-in and pit-out laps, safety car and virtual safety car laps (including the restart lap), laps slower than 107% of the driver's median, laps with missing values, drivers with fewer than 15 clean laps, stints with fewer than 8 clean laps. The count after each step is written to `metrics/preprocess_report.json`.

**Targets** (one row per clean lap).

- `pace_loss_s`: fuel-corrected lap time minus the 10th percentile of fuel-corrected lap times in the same stint. Fuel-corrected time is `lap_duration + 0.05 * lap_number`. The 0.05 s per lap is an estimate: the data gives 0.049 +/- 0.001 s per lap for fuel burn plus track evolution together (`python -m src.data.diagnose_target`), and it matches the physical prior.
- `pit_recommended`: true when the mean of `pace_loss_s` over the last 3 clean laps of the stint (including the current lap) is at least 1.0 s. This is a heuristic label for "tyres are costing real time", not a record of what teams decided. Rows without 3 laps of history in the stint are dropped.

**Features** (15, the model input).

| Feature | Meaning |
|---|---|
| `tyre_age` | Laps on the current set, including the age it had at the start of the stint |
| `compound` | SOFT, MEDIUM or HARD |
| `stint_number`, `lap_number` | Position in the race |
| `race_progress` | `lap_number / total race laps` |
| `track_temperature`, `air_temperature`, `humidity`, `rainfall`, `wind_speed` | Weather at the start of the lap |
| `track_temp_delta` | Track temperature now minus 5 laps ago (missing if that lap was removed) |
| `prev_lap_delta` | Fuel-corrected time of the previous clean lap minus the one before it, same stint |
| `pace_decay_slope` | Least-squares slope (s per lap) of fuel-corrected time over the previous 5 clean laps, same stint |
| `prev_gap_to_best` | Fuel-corrected time of the previous clean lap minus the best earlier lap of the stint |
| `recent_gap_to_best` | Mean fuel-corrected time of the previous 2 clean laps minus the same best lap |

The `gap_to_best` features are an online version of the pace-loss target: the target measures against the stint's 10th percentile lap (known only once the stint is over), these measure against the best lap seen so far. They exist because a persistence baseline built from true previous pace loss scored far above a model without them. The five lag features use only earlier laps of the same stint, never the current lap, so they can be supplied at prediction time. "Previous clean laps" means the driver's earlier laps after the filters above, so safety car, pit and outlier laps are skipped rather than counted. They are missing for the first laps of a stint, and models must accept missing values.

**Split.** By season, never random: train on 2023 and 2024, test on 2025. A random lap split would put laps from the same stint in both sets.

## Training and model registry

`python -m src.models.train` (or `dvc repro`) trains on 2023 and 2024 and evaluates once on 2025. One run produces two models of the same algorithm family and packages them as one.

1. **Regressor** (predicts `pace_loss_s`) and **classifier** (predicts the probability of `pit_recommended`). Candidates are a naive lookup baseline, Random Forest and LightGBM (15 random-search trials, Huber loss, which is robust to traffic spikes in lap times). The winner has the lowest out-of-fold MAE.
2. **Cross-validation holds out whole races** (`GroupKFold` by session), so laps of one race never sit on both sides of a fold.
3. **Pit threshold.** The probability cut-off that maximises F1 is chosen on out-of-fold predictions from the training years only, never on the test year.
4. **Gates** (`params.yaml`, section `train`). Gates and promotion use cross-validation on the training years only, so the 2025 test year informs no decision and is a clean report. The run fails, nothing is registered and the report is still written, if the regressor's CV MAE is not at least 5% better than the baseline (`min_lift_over_baseline`), or if the classifier's CV PR-AUC is below the compound-and-tyre-age lookup baseline (`min_pr_auc_lift_over_baseline`). An optional third gate, `min_pr_auc_lift_over_persistence` (null = report only), compares against the persistence baseline below.
5. **Registry.** Every successful run registers a new version in the local MLflow registry (`mlflow.db`). The alias `champion` moves to it if there is no champion, if its CV MAE is at least 1% better, or if it ran on exactly the same train and test files as the champion and is not worse. The third rule means a code change on unchanged data replaces the old model instead of leaving a stale one in `models/champion`. A worse model on the same data, or a marginal gain on new data, does not take the alias. The champion is exported to `models/champion`.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db     # compare runs at http://127.0.0.1:5000
```

Each MLflow run records the parameters, cross-validation and test metrics, permutation importance, single-row latency, the git commit and the MD5 of the train and test files. `metrics/train_metrics.json` holds the same summary: `cv_mae` and `baseline_test_mae` for the regressor, `test_classification`, `baseline_test_classification` and `persistence_test_classification` for the classifier, `cv_lift_over_baseline`, `pr_auc_lift_over_persistence`, `latency`, `top_features`, `promoted`, `model_version` (this run) and `champion_version` (what is exported).

**Persistence baseline.** The pit label is a trailing mean of pace loss, so a classifier could score well just by extrapolating a stint's recent slow laps. The persistence baseline scores each lap by the mean pace loss of the stint's previous 2 laps, with no training. `pr_auc_lift_over_persistence` in the report is the model's PR-AUC minus that baseline's. It is generous to the baseline, because pace loss uses the stint's best laps as reference, which a live service would not yet know. A model that does not clearly beat it is mostly restating the label's own history.

**Reproducibility.** Seeds live in `params.yaml`, folds are deterministic and LightGBM runs with its deterministic flags, so the same data, parameters and seed give the same model regardless of thread count. Package versions are recorded in the model's `requirements.txt`.

**Registry scope.** `mlflow.db` and `mlartifacts/` exist per machine and are not shared. What is shared is `models/champion`, through DVC.

## Using the model

```python
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models/champion")
row = pd.DataFrame([{
    "tyre_age": 18, "compound": "MEDIUM", "stint_number": 1, "lap_number": 22, "race_progress": 0.38,
    "track_temperature": 41.0, "air_temperature": 28.0, "humidity": 45.0, "rainfall": 0.0, "wind_speed": 2.5,
    "track_temp_delta": 1.2, "prev_lap_delta": 0.15, "pace_decay_slope": 0.08,
    "prev_gap_to_best": 0.2, "recent_gap_to_best": 0.35,
}])
model.predict(row)   # DataFrame with pace_loss_s, pit_probability, pit_recommended
```

- All 15 feature columns are required (see "Features"); a missing column raises `ValueError`. Extra columns are ignored. Whole numbers may be sent as integers.
- `compound` must be SOFT, MEDIUM or HARD (any case). Anything else, including INTERMEDIATE and WET, raises `ValueError`: the model is trained on dry races only, and a serving layer should answer with a 422 and not a guess.
- `track_temp_delta`, `prev_lap_delta`, `pace_decay_slope`, `prev_gap_to_best` and `recent_gap_to_best` may be missing (`None` or `NaN`). They are naturally missing early in a stint.
- The model is a pickle. Load it only from the team's own DVC remote, with the package versions in `models/champion/requirements.txt` and the same Python minor version it was trained on (3.11), for example a `python:3.11` Docker base image.

## Serving the model (Phase 3)

A FastAPI service in `src/api/` wraps `models/champion`. It needs the model on disk (`dvc pull`, or the exported folder copied into `models/champion`).

```bash
uvicorn src.api.app:app --port 8000          # local, from the repository root
docker compose up --build                    # container, same port; build after models/champion exists
```

Interactive docs are at `http://localhost:8000/docs`.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process is up. Always 200 |
| `GET /ready` | 200 once the model loaded and passed a smoke prediction, else 503 |
| `GET /model-info` | Threshold, feature list, champion version and test metrics from `metrics/train_metrics.json` |
| `POST /predict` | One lap, all 15 features (the four lag features and `track_temp_delta` may be null) |
| `POST /predict/batch` | `{"laps": [...]}`, same fields, up to 500 laps |
| `POST /predict/from-history` | Current-lap context plus the stint's earlier clean laps (`lap_number`, `lap_duration`); the service computes the four lag features with the preprocessing formulas |

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{
  "tyre_age": 18, "compound": "MEDIUM", "stint_number": 1, "lap_number": 22, "race_progress": 0.38,
  "track_temperature": 41, "air_temperature": 28, "humidity": 45, "wind_speed": 2.5,
  "prev_gap_to_best": 0.4, "recent_gap_to_best": 0.5}'
# {"pace_loss_s": ..., "pit_probability": ..., "pit_recommended": ..., "warnings": [], "threshold": ..., "request_id": "..."}
```

Behaviour worth knowing:

- Invalid input is a 422 that names the field and the reason, never the rejected value. Unknown field names, NaN or infinity, out of range numbers, wet compounds (`INTERMEDIATE`, `WET`) and unordered or future history laps are all rejected. A model that is not loaded gives 503.
- `rainfall > 0` still predicts but returns a warning, because the model was trained on dry races. Missing lag features also return a warning, because accuracy drops without them.
- `/predict/from-history` expects clean laps only (no pit-in or pit-out laps, safety car laps or laps slower than 107% of the driver's median) and fails if they are not in strictly increasing lap order. It reads `fuel_correction_s_per_lap`, `decay_window` and `recent_gap_window` from `params.yaml`, so the image must ship the same `params.yaml` the model was trained with. `tests/test_api_history.py` checks that this code and `preprocess.py` agree row by row.
- Every prediction is appended to a JSON-lines file (`PREDICTION_LOG`, default `logs/predictions.jsonl`, a named volume in Compose) with timestamp, request id, inputs, outputs and latency. An empty `PREDICTION_LOG` disables it. A logging failure never fails a request. Phase 4 drift monitoring reads this file.
- Configuration is by environment variable: `MODEL_DIR`, `METRICS_PATH`, `PARAMS_PATH`, `PREDICTION_LOG`, `MAX_BATCH`.
- There is no authentication, rate limiting or request body size limit. Put the service behind a gateway or reverse proxy before exposing it beyond a trusted network.
- The Docker image installs the exact package versions recorded in `models/champion/requirements.txt` on `python:3.11-slim`, so the model runs on the Python and library versions that trained it. Rebuild the image whenever `models/champion` changes. Serving the model from a repository checkout at a different commit than the one that exported it is unsupported, because the model's pickle references the `src` modules.

## Automation and monitoring (Phase 4)

**CI** (`.github/workflows/ci.yml`) runs on every pull request and every push to `main`:
`ruff` and `pytest` (job `test`), then a Docker smoke test (job `docker`) and the Airflow DAG tests (job `airflow-dag`). The smoke test trains a throwaway synthetic champion with `python -m tests.make_ci_champion`, builds the image, starts it, waits for `/ready`, checks a prediction and checks that a wet compound is refused. It proves the Dockerfile, API and model contract fit together; it says nothing about the real model, which CI cannot reach while the DVC remote is a folder on one laptop. `publish-image.yml` publishes the image with the real model to GHCR, is manual only, and needs a shared DVC remote and repository secrets first (see its header). It has not been exercised.

**Drift** (`python -m src.monitoring.drift`) compares the served requests in `logs/predictions.jsonl` with `data/processed/train.parquet` using Evidently and writes `reports/drift/drift_report.html` and `drift_summary.json`.

- Only input drift is measured. Labels (pace loss, pit flag) exist only after a stint ends, so live model performance cannot be monitored; shifted inputs are the early warning.
- Fewer than 200 logged requests gives `status: insufficient_data` and no drift claim.
- A column drifts when its normalised Wasserstein distance is 0.2 or more (numeric) or its Jensen-Shannon distance is 0.1 or more (compound). Evidently's default of 0.1 flagged columns that had not moved at a few hundred rows. Overall drift is declared when 20% of columns drift (three of fourteen), because the weather columns move together. With identical distributions no overall drift appeared in 15 of 15 trials at 200 rows. These are starting values; tune `--num-threshold`, `--cat-threshold` and `--drift-share` once there is real traffic.
- Constant columns (`rainfall` is 0 throughout training) are skipped and listed in the summary. `--fail-on-drift` makes the exit status 2 on drift.

**Retraining DAG** (`airflow/dags/f1_retrain.py`, weekly). `choose_path` goes straight to `ingest` when the run parameter `force_retrain` is set, otherwise `check_drift` runs and `decide_after_drift` either skips or continues with `ingest`, `preprocess`, `train` and `summarize`. Tasks call the same `python -m ...` commands as `dvc repro`, inside the project's own virtualenv (`F1_PYTHON`, `F1_PROJECT_DIR`), so Airflow's dependency pins never meet the project's. A missing drift summary fails the run instead of guessing; `ingest` retries twice, `train` never retries, and a quality-gate rejection fails the DAG and keeps the old champion. The DAG does not commit anything: after a retrain, review the result, run `dvc status`, then commit `dvc.lock` and `metrics/` and `dvc push`. The API container must be rebuilt and restarted to serve a new champion.

```bash
echo "AIRFLOW_UID=$(id -u)" > .env                               # once, Linux
docker compose up --build -d                                  # API only, http://localhost:8000
docker compose --profile airflow up --build -d                # API + Airflow, http://localhost:8080
python -m pytest -q tests && python -m pytest -q airflow/tests   # the second needs apache-airflow installed
```

`docker compose down` stops everything; the prediction log and Airflow state live in named volumes and survive it (`down -v` deletes them). Add `COMPOSE_PROFILES=airflow` to `.env` to make a plain `docker compose up` start both.

The Airflow image was built and its login page served on the author's machine; a full DAG run inside the container has not been done; the DAG itself was parsed and run through its drift branch with Airflow 2.10.5.

## Repository layout

```
params.yaml              all tunable values (ingest, preprocess)
dvc.yaml  dvc.lock       pipeline definition and the exact hashes of the last run
src/config.py            paths and params loading
src/data/ingest.py       OpenF1 download with retries
src/data/preprocess.py   cleaning, targets, features, split
src/data/diagnose_target.py   compares target definitions, estimates the fuel effect
src/features.py          FeaturePreparer: the one place that turns rows into model input
src/models/estimators.py baselines, Random Forest, LightGBM, search space
src/models/evaluate.py   metrics, F1 threshold, single-row latency
src/models/bundle.py     the deployable model (regressor + classifier + threshold)
src/models/train.py      comparison, gates, MLflow logging, registry, export
src/api/                 FastAPI service: schemas, model service, history features, prediction log
Dockerfile  docker-compose.yml  requirements-api.txt   serving image
src/monitoring/drift.py  Evidently drift report over the prediction log
airflow/                 retrain DAG, Airflow image, DAG tests
.github/workflows/       CI and the manual image publish
tests/                   pytest suite on synthetic OpenF1-shaped data
metrics/                 small JSON reports, tracked by Git
models/champion          the exported model, tracked by DVC
data/                    tracked by DVC, never committed to Git
mlflow.db  mlartifacts/  local MLflow registry (gitignored, one per machine)
```

## Team workflow

- `main` is protected: no direct pushes. All work happens on branches named `<name>/<topic>`, merged through a reviewed pull request.
- Stacked pull requests (a branch built on another unmerged branch): merge the lower one first using "Create a merge commit". Squash or rebase merges rewrite the lower commit and make the upper pull request show it again.
- Data and model files go to DVC (`dvc push`), only `dvc.lock`, `dvc.yaml` and small metrics files go to Git. After pulling, run `dvc pull`.
- Do not edit a file that a DVC stage depends on without rerunning `dvc repro` and committing the updated `dvc.lock`.

**Sharing data.** The DVC remote is a local folder (`/home/saad/Documents/dvc-storage`), so a teammate can only `dvc pull` if they can read a copy of that folder. Either copy it to a shared drive and point `dvc remote modify --local localstore url <path>` at it, or switch to a shared remote later.

## Troubleshooting

- `ssh: connect to host github.com port 22: Connection timed out`: the network blocks SSH. Use SSH over port 443 by adding `Host github.com`, `Hostname ssh.github.com`, `Port 443`, `User git` to `~/.ssh/config`.
- `dvc repro` runs `ingest` again after a code change: it requests the session lists, skips sessions already on disk and retries the empty ones.
- `Session NNNN incomplete, skipping`: see the ingestion notes above.

## Limits

- Dry races only; wet-weather strategy is out of scope.
- The pit flag is a heuristic derived from pace loss, not ground truth about team decisions. It ignores traffic, gaps, safety car timing and strategy.
- Season 2025 is only a report: champion selection and gates use cross-validation on 2023 and 2024. It is still the single test year, so it is one sample of conditions. Automated retraining needs a rolling holdout.
- Pace loss has a large noise floor (traffic, lapping, track position), so the regressor explains a minority of the variance. Judge it against the baseline in `metrics/train_metrics.json`, not by R2 alone. The pit flag is the more useful output.
- The fuel and track-evolution correction (0.05 s per lap) is an estimate, and tyre degradation estimates depend on it. Absolute pace-loss values carry that uncertainty.
