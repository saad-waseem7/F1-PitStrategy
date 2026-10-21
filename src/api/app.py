"""FastAPI application: `uvicorn src.api.app:app`."""
from __future__ import annotations

import logging
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.history import history_features
from src.api.prediction_log import PredictionLog
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HistoryRequest,
    HistoryResponse,
    LapFeatures,
    ModelInfo,
    Prediction,
    PredictionResponse,
)
from src.api import ui
from src.api.service import ModelNotReady, ModelService
from src.api.settings import Settings

LOG = logging.getLogger(__name__)

NO_HISTORY_WARNING = "No lap history supplied: the lag features are missing and predictions are noticeably less accurate."
WET_WARNING = "rainfall > 0: the model was trained on dry races only, so this prediction is unreliable."


def _warnings(lap: LapFeatures) -> list[str]:
    out = []
    if lap.rainfall > 0:
        out.append(WET_WARNING)
    if lap.prev_gap_to_best is None and lap.recent_gap_to_best is None:
        out.append(NO_HISTORY_WARNING)
    return out


def _to_predictions(frame: pd.DataFrame, laps: list[LapFeatures]) -> list[Prediction]:
    return [
        Prediction(
            pace_loss_s=float(row.pace_loss_s),
            pit_probability=float(row.pit_probability),
            pit_recommended=bool(row.pit_recommended),
            warnings=_warnings(lap),
        )
        for row, lap in zip(frame.itertuples(index=False), laps, strict=True)
    ]


def create_app(settings: Settings | None = None, service: ModelService | None = None) -> FastAPI:
    cfg = settings or Settings.from_env()
    svc = service or ModelService(cfg.model_dir, cfg.metrics_path, cfg.params_path)
    plog = PredictionLog(cfg.prediction_log)
    replay = ui.load_replay_store(cfg)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            svc.load()
            LOG.info("Model loaded from %s", svc.model_dir)
        except Exception:
            # Stay up so /health answers and /ready reports the problem, instead of crash looping silently.
            LOG.exception("Model failed to load from %s", svc.model_dir)
        yield

    app = FastAPI(
        title="F1 Pit Strategy",
        version="1.0.0",
        description="Predicts tyre-wear pace loss and a pit recommendation for dry F1 races.",
        lifespan=lifespan,
    )
    app.state.service = svc

    @app.middleware("http")
    async def request_id(request: Request, call_next: Any) -> Any:
        request.state.request_id = uuid.uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
        # The default handler echoes the rejected input, which breaks JSON encoding for NaN and
        # reflects arbitrary client data back. Report only where and why.
        detail = [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    @app.exception_handler(ModelNotReady)
    async def _not_ready(_: Request, exc: ModelNotReady) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": "Model is not loaded yet or failed to load."})

    def _log(request: Request, endpoint: str, lap: LapFeatures, pred: Prediction, latency_ms: float) -> None:
        plog.write({
            "request_id": request.state.request_id, "endpoint": endpoint, "inputs": lap.model_dump(),
            "outputs": pred.model_dump(exclude={"warnings"}), "latency_ms": latency_ms,
        })

    def _ms(start: float) -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    def _run(laps: list[LapFeatures]) -> list[Prediction]:
        frame = pd.DataFrame([lap.model_dump() for lap in laps])
        try:
            return _to_predictions(svc.predict(frame), laps)
        except ValueError as exc:
            # The model raises ValueError for inputs it refuses (wet compound, missing column).
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health", tags=["ops"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready", tags=["ops"])
    def ready() -> dict[str, bool]:
        if not svc.ready:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"ready": True}

    @app.get("/model-info", response_model=ModelInfo, tags=["ops"])
    def model_info() -> dict[str, Any]:
        return svc.info()

    @app.post("/predict", response_model=PredictionResponse, tags=["predict"])
    def predict(lap: LapFeatures, request: Request) -> PredictionResponse:
        start = time.perf_counter()
        pred = _run([lap])[0]
        _log(request, "/predict", lap, pred, _ms(start))
        return PredictionResponse(**pred.model_dump(), threshold=svc.threshold, request_id=request.state.request_id)

    @app.post("/predict/batch", response_model=BatchResponse, tags=["predict"])
    def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
        if len(body.laps) > cfg.max_batch:
            raise HTTPException(status_code=413, detail=f"batch too large: {len(body.laps)} > {cfg.max_batch}")
        start = time.perf_counter()
        preds = _run(body.laps)
        elapsed = _ms(start)
        for lap, pred in zip(body.laps, preds, strict=True):
            _log(request, "/predict/batch", lap, pred, elapsed)
        return BatchResponse(predictions=preds, threshold=svc.threshold, request_id=request.state.request_id)

    @app.post("/predict/from-history", response_model=HistoryResponse, tags=["predict"])
    def predict_from_history(body: HistoryRequest, request: Request) -> HistoryResponse:
        """Send the stint's earlier clean laps and the service computes the four lag features itself."""
        p = svc.preprocess_params
        if not p:
            raise ModelNotReady("model is not loaded")
        try:
            feats = history_features(
                [lap.lap_number for lap in body.previous_laps],
                [lap.lap_duration for lap in body.previous_laps],
                body.lap_number,
                float(p["fuel_correction_s_per_lap"]),
                int(p["decay_window"]),
                int(p["recent_gap_window"]),
            )
            lap = LapFeatures(**body.model_dump(exclude={"previous_laps"}), **feats)
        except ValueError as exc:  # unordered laps, or derived features outside the accepted ranges
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        start = time.perf_counter()
        pred = _run([lap])[0]
        _log(request, "/predict/from-history", lap, pred, _ms(start))
        return HistoryResponse(
            **pred.model_dump(), threshold=svc.threshold, request_id=request.state.request_id,
            computed_features=feats,
        )

    ui.register(app, cfg, svc, replay)
    return app


app = create_app()
