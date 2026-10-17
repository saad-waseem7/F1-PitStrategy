# Serving image for the champion model. Build from the repository root AFTER `dvc pull`
# (or after copying models/champion in), because the model is not in Git.
#   docker build -t f1-pit-strategy .
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /build
# Dependency layers first so code changes do not reinstall packages.
COPY models/champion/requirements.txt model-requirements.txt
COPY requirements-api.txt .
RUN pip install -r model-requirements.txt -r requirements-api.txt

FROM python:3.11-slim
# LightGBM needs the OpenMP runtime, which the slim image lacks.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    MODEL_DIR=/app/models/champion \
    METRICS_PATH=/app/metrics/train_metrics.json \
    PARAMS_PATH=/app/params.yaml \
    PREDICTION_LOG=/app/logs/predictions.jsonl
WORKDIR /app
COPY src ./src
COPY params.yaml ./
COPY metrics/train_metrics.json ./metrics/train_metrics.json
COPY models/champion ./models/champion
RUN mkdir -p /app/logs && chown -R app:app /app/logs
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).status == 200 else 1)"
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
