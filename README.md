# F1-PitStrategy

Predicts tyre-wear pace loss and whether a pit stop should be considered, using Formula 1 lap, stint, and weather data from OpenF1.

## Included in this phase

- OpenF1 ingestion and preprocessing for dry races.
- Pace-loss and heuristic pit recommendation targets, with features based on earlier laps.
- Regression and classification baselines, Random Forest, and LightGBM.
- Grouped cross-validation by race session and a season-based test split.
- MLflow tracking, model registration, and export of the champion model.
- DVC stages for data and model files.

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

Run commands from the repository root. Use `python -m src.models.train` to run training directly, or `dvc repro` to run the pipeline.

## Evaluation

Training uses 2023 and 2024, with 2025 held out for testing. The regressor predicts pace loss in seconds, and the classifier predicts the probability of the heuristic pit label. Check `metrics/train_metrics.json` for the latest results.

The classifier does not beat the recent-lap persistence baseline in the current reported test results. This is recorded in the metrics and should be considered when judging the model.

## Data and model notes

- Wet and mixed-weather races are out of scope.
- Pit labels are heuristic, not ground truth about team decisions.
- Data and the exported champion model are managed by DVC.
- MLflow runs and the local model registry are not shared automatically.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy or a shared remote before `dvc pull` will work.
