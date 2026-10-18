# F1-PitStrategy

Predicts tyre-wear pace loss and whether a pit stop should be considered, using Formula 1 lap, stint, and weather data from OpenF1.

## Included in this branch

- OpenF1 ingestion and preprocessing for dry races.
- Model training, evaluation, and MLflow tracking.
- A FastAPI service for single-lap, batch, and history-based predictions.
- Docker and Docker Compose setup for running the API.
- Request validation and a log of served predictions.

## Setup

Requirements: Python 3.11, Git, and Docker for container use.

```bash
git clone git@github.com:saad-waseem7/F1-PitStrategy.git
cd F1-PitStrategy
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
ruff check src tests
```

The API needs the exported model in `models/champion`. Get it with `dvc pull` if you have access to the configured DVC remote.

## Run the API

From the repository root:

```bash
uvicorn src.api.app:app --port 8000
```

Or run it in Docker after the model is available:

```bash
docker compose up --build
```

Open `http://localhost:8000/docs` for the interactive API documentation.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Checks that the service is running |
| `GET /ready` | Checks that the model loaded successfully |
| `GET /model-info` | Shows model details and reported metrics |
| `POST /predict` | Predicts for one lap |
| `POST /predict/batch` | Predicts for a batch of laps |
| `POST /predict/from-history` | Calculates lag features from earlier clean laps, then predicts |

Invalid input returns a validation error. Wet tyre compounds are rejected because the model was trained on dry races. Rainfall and missing history features can produce warnings. Served predictions are logged to `logs/predictions.jsonl` for monitoring.

## Data and model notes

- The pit recommendation is a heuristic, not a record of actual team decisions.
- The model is stored with DVC and must be kept with the code and package versions used to export it.
- The configured DVC remote is a local folder (`/home/saad/Documents/dvc-storage`). Teammates need access to a shared copy or a shared remote.
- The API has no authentication or rate limiting. Do not expose it directly to an untrusted network.
