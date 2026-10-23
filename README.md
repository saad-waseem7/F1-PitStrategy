# F1-PitStrategy

A tyre-wear and pit-stop predictor for Formula 1, built as the CS4085 MLOps project at FAST-NUCES Lahore. Give it a lap (tyre age, compound, weather, how far into the race) and it estimates how many seconds the tyres are costing, then says whether to box or stay out.

It covers data ingestion from OpenF1, a DVC pipeline, MLflow-tracked training and a FastAPI service. Only dry races are supported, and the pit call is a rule of thumb, not real team strategy.

## How well does it work?

Trained on 2023 and 2024, tested on 2025 (18,373 laps).

| | Naive baseline | LightGBM (champion) |
|---|---|---|
| Pace loss error (MAE, seconds) | 0.531 | **0.375** |
| Pit flag PR-AUC | 0.448 | **0.884** |

The baseline just looks up the typical value for a compound and tyre age. LightGBM cuts the error by about 29% against it. R² is 0.49, so it is useful but not sharp. One caveat: a baseline that reuses the actual pace loss of the previous laps scores higher on the pit flag (PR-AUC 0.951). It reads information a live service would not have, so we report it instead of chasing it.

## Run it

You need Python 3.11, Git and DVC, plus Docker for the app.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

The data and the trained model (`models/champion`) are tracked by DVC, not Git. The DVC remote in `.dvc/config` is a folder on the author's laptop, so `dvc pull` only works there. Anyone else has two options: run `dvc repro` (slow, because OpenF1 is throttled, and numbers can shift a little if OpenF1 changes its data), or point DVC at storage of your own with `dvc remote modify localstore url <path>`.

With the model in place:

```bash
docker compose up --build -d
```

Then open `http://localhost:8000/` and `http://localhost:8000/docs` for the API docs.

## How it works

| Stage | Command | Output |
|---|---|---|
| ingest | `python -m src.data.ingest` | `data/raw/` |
| preprocess | `python -m src.data.preprocess` | `data/processed/` |
| train | `python -m src.models.train` | `models/champion/` |

`dvc repro` runs them in order, and every setting lives in `params.yaml`.

We start with 77,075 raw laps from 61 dry race sessions and keep 51,833 after dropping lap 1, pit laps, safety car laps, wet sessions and slow outliers. Pace loss is a lap's fuel-corrected time minus the stint's reference pace. The pit flag is set when the average pace loss over the last 3 laps reaches 1 second, which is a heuristic and not a real team decision.

There are 15 features: tyre age, compound, stint and lap number, race progress, the weather, and four lag features built only from earlier laps of the same stint (recent lap change, pace decay slope, and the gap to the best lap so far). The lag features are missing early in a stint.

Training compares a naive baseline, a random forest and LightGBM with 5-fold cross-validation grouped by race, so no race appears on both sides. The run fails on purpose if the model is not at least 5% better than the baseline, and a new model only replaces the champion if it is clearly better. Runs and model versions go to MLflow.

## The API

`POST /predict` scores one lap. `POST /predict/from-history` also takes the stint's earlier lap times and works out the lag features for you. `POST /predict/batch` scores many laps at once (500 by default). `GET /health`, `/ready` and `/model-info` cover status, the loaded model and its held-out metrics.

```bash
curl -s localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "tyre_age": 14, "compound": "MEDIUM", "stint_number": 1, "lap_number": 18,
  "race_progress": 0.316, "track_temperature": 38, "air_temperature": 27,
  "humidity": 45, "wind_speed": 2.5
}'
```

You get back `pace_loss_s`, `pit_probability`, `pit_recommended` and a `request_id`. Inputs are validated strictly, so bad values, unknown fields and wet compounds come back as a 422 that names the field. Every prediction is appended to a log file, which the drift check reads. Settings come from environment variables such as `MODEL_DIR`, `PREDICTION_LOG` and `MAX_BATCH` (see `src/api/settings.py`).

## Project layout

```
src/data/        ingest and preprocess
src/features.py  one shared feature preparer for training and serving
src/models/      estimators, evaluation, training, model export
src/api/         FastAPI app, schemas, model service
tests/           tests on synthetic data
params.yaml      every pipeline setting
dvc.yaml         the pipeline
```

## Tests

`python -m pytest -q` runs the tests on synthetic data, with no network needed.

## Limitations

- Dry races only. Wet sessions are excluded from training and wet compounds are rejected.
- The pit flag is a rule on recent pace loss, so precision and recall only measure agreement with that rule.
- Pace loss R² is moderate, and predictions without earlier laps are noticeably weaker.
- No authentication or rate limiting. Keep it on a trusted network.
- The data and model need DVC, whose remote is a local folder, so a plain clone cannot run the app.
