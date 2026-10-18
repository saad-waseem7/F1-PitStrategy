# F1-PitStrategy

Predicts tyre-wear pace loss and whether a pit stop should be considered, using Formula 1 lap, stint and weather data from OpenF1. This project is being built in phases for the CS4085 MLOps course.

## What is included on this branch

- OpenF1 data ingestion with retries and resumable downloads.
- Preprocessing for dry races, pace-loss and pit recommendation targets, lag features, and a season-based train/test split.
- Regression and classification models, grouped cross-validation, MLflow tracking, model registration, and export of the champion model.
- DVC pipeline and data/model versioning.

The API and monitoring work is being developed on separate branches and pull requests.

## Setup

Requirements: Python 3.11, Git, and DVC.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest -q
ruff check src tests
dvc pull
dvc repro
```

Run commands from the repository root. The code imports modules through `src.*`.

## Pipeline

```text
OpenF1 -> data/raw/ -> preprocessing -> data/processed/
                                  -> training -> models/champion/
```

Run individual stages with `python -m src.data.ingest`, `python -m src.data.preprocess`, and `python -m src.models.train`. The same stages can be run through `dvc repro`.

## Data and model notes

- The model is trained on dry races. Wet and mixed-weather sessions are excluded.
- The pit recommendation is a heuristic based on recent pace loss, not a record of real team decisions.
- Training uses 2023 and 2024, with 2025 held out for testing.
- `metrics/` contains small reports tracked by Git. Data and the exported model are managed by DVC.
- MLflow runs and its local registry are stored on each machine and are not shared automatically.

## Sharing data

The configured DVC remote is a local folder: `/home/saad/Documents/dvc-storage`. Teammates need access to a shared copy of that folder, or the project needs a shared remote, before `dvc pull` will work for them.

## Tests

The test suite uses synthetic OpenF1-shaped data and does not need to download race data. Run `python -m pytest -q` from the repository root.

## Known limits

The model only covers dry races. Pit labels are heuristic, and lap times are affected by traffic, track position, and other factors that are not fully represented. The fuel correction is an estimate, so pace-loss values also carry uncertainty.
