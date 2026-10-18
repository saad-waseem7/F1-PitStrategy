"""The web page and the read-only endpoints behind it (stint replay, drift summary)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.monitoring import read_drift_summary
from src.api.schemas import DriftSummary, ReplayLap, ReplaySessions, ReplayStint
from src.api.service import ModelService
from src.api.settings import Settings
from src.config import FEATURE_COLUMNS
from src.replay.store import ReplayError, ReplayStore

LOG = logging.getLogger(__name__)

# Scripts and styles only from this origin, no framing. Applied to the page and its assets, not to /docs,
# whose Swagger UI needs inline scripts.
UI_SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


def is_ui_path(path: str) -> bool:
    return path == "/" or path.startswith("/assets/")


def load_replay_store(cfg: Settings) -> ReplayStore:
    store = ReplayStore(cfg.replay_path)
    try:
        store.load()
        LOG.info("Replay sample loaded from %s", cfg.replay_path)
    except ReplayError as exc:
        LOG.warning("Replay disabled: %s", exc)
    return store


def register(app: FastAPI, cfg: Settings, svc: ModelService, replay: ReplayStore) -> None:
    index = cfg.web_dir / "index.html"
    assets = cfg.web_dir / "assets"

    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/", include_in_schema=False)
    def home() -> FileResponse:
        if not index.is_file():
            raise HTTPException(status_code=404, detail="web page not installed")
        return FileResponse(index, media_type="text/html")

    @app.get("/replay/sessions", response_model=ReplaySessions, tags=["replay"])
    def replay_sessions() -> dict[str, Any]:
        if not replay.available:
            return {"available": False}
        return {"available": True, "meta": replay.meta, "sessions": replay.sessions()}

    @app.get("/replay/stint", response_model=ReplayStint, tags=["replay"])
    def replay_stint(
        session_key: int = Query(ge=0),
        driver_number: int = Query(ge=0),
        stint_number: int = Query(ge=0),
    ) -> ReplayStint:
        """Model output for every lap of one stint of a sampled race, next to the true labels.

        These calls are NOT written to the prediction log: replayed laps are not served traffic and would
        distort the drift check.
        """
        rows = replay.stint(session_key, driver_number, stint_number)
        if rows is None:
            raise HTTPException(status_code=404, detail="stint not in the replay sample")
        preds = svc.predict(rows[list(FEATURE_COLUMNS)])
        laps = [
            ReplayLap(
                lap_number=int(row.lap_number),
                tyre_age=float(row.tyre_age),
                compound=str(row.compound),
                actual_pace_loss_s=float(row.pace_loss_s),
                actual_pit_recommended=bool(row.pit_recommended),
                predicted_pace_loss_s=float(p.pace_loss_s),
                pit_probability=float(p.pit_probability),
                pit_recommended=bool(p.pit_recommended),
            )
            for row, p in zip(rows.itertuples(index=False), preds.itertuples(index=False), strict=True)
        ]
        return ReplayStint(
            session_key=session_key, driver_number=driver_number, stint_number=stint_number,
            circuit=str(rows["circuit_short_name"].iloc[0]), year=int(rows["year"].iloc[0]),
            threshold=svc.threshold, laps=laps,
        )

    @app.get("/monitoring/summary", response_model=DriftSummary, tags=["monitoring"])
    def monitoring_summary() -> dict[str, Any]:
        return read_drift_summary(cfg.drift_summary_path)

    @app.middleware("http")
    async def ui_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if is_ui_path(request.url.path):
            for name, value in UI_SECURITY_HEADERS.items():
                response.headers.setdefault(name, value)
        return response
