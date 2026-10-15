# F1-PitStrategy

Predicts how much lap time a Formula 1 car is losing to tyre wear (seconds) and whether the team should consider pitting (true/false), from OpenF1 lap, stint and weather data. Built for CS4085 MLOps (FAST-NUCES Lahore, Fall 2026).

This repository is built in phases. This README describes what is implemented on `main`; sections for later phases are added by the people who own them.

| Phase | Owner | Content |
|---|---|---|
| 1 | Saad | Data ingestion, preprocessing, DVC pipeline |
| 2 | Saad | Model training, MLflow tracking and registry |
| 3 | Partner | FastAPI service, Docker, Docker Compose |
| 4 | Partner | GitHub Actions CI/CD, Evidently drift checks, Airflow DAGs |

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
```

Stages are defined in `dvc.yaml`; all tunable values live in `params.yaml`.

| Stage | Command | Output |
|---|---|---|
| `ingest` | `python -m src.data.ingest` | `data/raw/` (laps, stints, weather, race_control, sessions) |
| `preprocess` | `python -m src.data.preprocess` | `data/processed/`, `metrics/preprocess_report.json` |

Ingestion is resumable: a session whose `laps` file exists is skipped, files are written atomically, and `persist: true` in `dvc.yaml` stops DVC from deleting `data/raw` before a rerun. Throttled, failed, truncated or empty responses are retried after a fixed 3 seconds (a longer `Retry-After` from the server is honoured). After the retries are used up the run fails loudly instead of producing a partial dataset.

A session for which OpenF1 returns no laps, stints and weather even after the retries is skipped with a warning (observed for session 9086). It is retried on every ingest run, which costs about 40 seconds each time, so it is safe to ignore.

## Data definition

**Scope.** Dry races only. A session is dropped if 5% or more of its laps are on wet tyres or in rain, and any wet-tyre lap left in a dry session is removed. The model is not valid for wet or mixed conditions.

**Laps removed.** Lap 1, pit-in and pit-out laps, safety car and virtual safety car laps (including the restart lap), laps slower than 107% of the driver's median, laps with missing values, drivers with fewer than 15 clean laps, stints with fewer than 8 clean laps. The count after each step is written to `metrics/preprocess_report.json`.

**Targets** (one row per clean lap).

- `pace_loss_s`: fuel-corrected lap time minus the 10th percentile of fuel-corrected lap times in the same stint. Fuel-corrected time is `lap_duration + 0.05 * lap_number`. The 0.05 s per lap is an estimate: the data gives 0.049 +/- 0.001 s per lap for fuel burn plus track evolution together (`python -m src.data.diagnose_target`), and it matches the physical prior.
- `pit_recommended`: true when the mean of `pace_loss_s` over the last 3 clean laps of the stint (including the current lap) is at least 1.0 s. This is a heuristic label for "tyres are costing real time", not a record of what teams decided. Rows without 3 laps of history in the stint are dropped.

**Features** (13, the model input).

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

The three lag features use only earlier laps of the same stint, never the current lap, so they can be supplied at prediction time. "Previous clean laps" means the driver's earlier laps after the filters above, so safety car, pit and outlier laps are skipped rather than counted. They are missing for the first laps of a stint, and models must accept missing values.

**Split.** By season, never random: train on 2023 and 2024, test on 2025. A random lap split would put laps from the same stint in both sets.

## Repository layout

```
params.yaml              all tunable values (ingest, preprocess)
dvc.yaml  dvc.lock       pipeline definition and the exact hashes of the last run
src/config.py            paths and params loading
src/data/ingest.py       OpenF1 download with retries
src/data/preprocess.py   cleaning, targets, features, split
src/data/diagnose_target.py   compares target definitions, estimates the fuel effect
tests/                   pytest suite on synthetic OpenF1-shaped data
metrics/                 small JSON reports, tracked by Git
data/                    tracked by DVC, never committed to Git
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
- The fuel and track-evolution correction (0.05 s per lap) is an estimate, and tyre degradation estimates depend on it. Absolute pace-loss values carry that uncertainty.
