# F1-PitStrategy

This branch builds the data pipeline for predicting tyre-wear pace loss and a pit recommendation from Formula 1 data.

## Included in this phase

- Downloads lap, stint, weather, race-control, and session data from OpenF1.
- Saves raw data by session and retries failed or incomplete requests.
- Removes unsuitable laps and wet or mixed-weather sessions.
- Builds pace-loss and heuristic pit recommendation targets, plus past-lap features.
- Splits the data by season: 2023 and 2024 for training, 2025 for testing.
- Tracks the pipeline and data with DVC.

## Setup

Requirements: Python 3.11, Git, and DVC.

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

Run commands from the repository root. To run the configured pipeline, use `dvc repro`.

## Data notes

- The pipeline is designed for dry races. Wet and mixed-weather sessions are excluded.
- The pit recommendation is a heuristic based on recent pace loss, not actual team strategy.
- Data files are managed by DVC and are not committed directly to Git.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy before they can use `dvc pull`.

## Tests

Run `python -m pytest -q`. Tests use synthetic data and do not need network access.
