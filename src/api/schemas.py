"""Request and response models. These are the public contract of the service."""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import COMPOUNDS
from src.features import WET_COMPOUNDS

DRY_COMPOUNDS: tuple[str, ...] = tuple(c for c in COMPOUNDS if c not in WET_COMPOUNDS)

Finite = dict(allow_inf_nan=False)


class _Strict(BaseModel):
    # A misspelled field name must be an error, not a silently ignored field.
    model_config = ConfigDict(extra="forbid")


class LapContext(_Strict):
    """What a caller knows about the current lap without any lap history."""

    tyre_age: Annotated[float, Field(ge=0, le=100, description="Laps on this set, including its age at stint start", **Finite)]
    compound: str = Field(description="SOFT, MEDIUM or HARD (any case). Wet compounds are out of scope.")
    stint_number: Annotated[int, Field(ge=1, le=10)]
    lap_number: Annotated[int, Field(ge=1, le=120)]
    race_progress: Annotated[float, Field(ge=0, le=1, description="lap_number / total race laps", **Finite)]
    track_temperature: Annotated[float, Field(ge=-10, le=90, description="degrees C", **Finite)]
    air_temperature: Annotated[float, Field(ge=-30, le=60, description="degrees C", **Finite)]
    humidity: Annotated[float, Field(ge=0, le=100, description="percent", **Finite)]
    rainfall: Annotated[float, Field(ge=0, le=1, **Finite)] = 0.0
    wind_speed: Annotated[float, Field(ge=0, le=60, description="m/s", **Finite)]
    track_temp_delta: Annotated[float | None, Field(ge=-40, le=40, description="Track temperature now minus 5 laps ago", **Finite)] = None

    @field_validator("compound", mode="before")
    @classmethod
    def _dry_compound(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("compound must be a string")
        upper = value.strip().upper()
        if upper in WET_COMPOUNDS:
            raise ValueError(f"{upper} is out of scope: the model supports dry races only")
        if upper not in DRY_COMPOUNDS:
            raise ValueError(f"compound must be one of {', '.join(DRY_COMPOUNDS)}")
        return upper


class LapFeatures(LapContext):
    """The full 15-feature model input. The four lag features may be null early in a stint."""

    prev_lap_delta: Annotated[float | None, Field(ge=-60, le=60, **Finite)] = None
    pace_decay_slope: Annotated[float | None, Field(ge=-10, le=10, **Finite)] = None
    prev_gap_to_best: Annotated[float | None, Field(ge=-1e-6, le=120, **Finite)] = None
    recent_gap_to_best: Annotated[float | None, Field(ge=-1e-6, le=120, **Finite)] = None


class LapRecord(_Strict):
    """One earlier clean lap of the current stint."""

    lap_number: Annotated[int, Field(ge=1, le=120)]
    lap_duration: Annotated[float, Field(gt=20, lt=300, description="Seconds, as reported by OpenF1", **Finite)]


class HistoryRequest(LapContext):
    previous_laps: list[LapRecord] = Field(
        default_factory=list,
        max_length=100,
        description=(
            "Earlier CLEAN laps of the current stint, oldest first: no pit-in or pit-out laps, "
            "safety car laps or laps slower than 107% of the driver's median."
        ),
    )


class BatchRequest(_Strict):
    laps: list[LapFeatures] = Field(min_length=1)


class Prediction(BaseModel):
    pace_loss_s: float
    pit_probability: float
    pit_recommended: bool
    warnings: list[str] = Field(default_factory=list)


class PredictionResponse(Prediction):
    threshold: float
    request_id: str


class HistoryResponse(PredictionResponse):
    computed_features: dict[str, float | None]


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    threshold: float
    request_id: str


class ModelInfo(BaseModel):
    ready: bool
    model_dir: str
    threshold: float | None = None
    features: list[str]
    mlflow_model_uuid: str | None = None
    created_utc: str | None = None
    champion_version: str | None = None
    algorithm: str | None = None
    test_metrics: dict[str, Any] = Field(default_factory=dict)


class ReplayStintInfo(BaseModel):
    stint_number: int
    compound: str
    laps: int
    first_lap: int
    last_lap: int


class ReplayDriver(BaseModel):
    driver_number: int
    stints: list[ReplayStintInfo]


class ReplaySession(BaseModel):
    session_key: int
    circuit: str
    year: int
    drivers: list[ReplayDriver]


class ReplaySessions(BaseModel):
    available: bool
    meta: dict[str, Any] = Field(default_factory=dict)
    sessions: list[ReplaySession] = Field(default_factory=list)


class ReplayLap(BaseModel):
    lap_number: int
    tyre_age: float
    compound: str
    actual_pace_loss_s: float
    actual_pit_recommended: bool
    predicted_pace_loss_s: float
    pit_probability: float
    pit_recommended: bool


class ReplayStint(BaseModel):
    session_key: int
    driver_number: int
    stint_number: int
    circuit: str
    year: int
    threshold: float
    laps: list[ReplayLap]


class DriftSummary(BaseModel):
    """The drift check's latest result. `status` is also "no_report" or "unreadable" when there is none."""

    status: str
    generated_at: str | None = None
    drift_detected: bool | None = None
    drifted_columns: list[str] = Field(default_factory=list)
    share_drifted: float | None = None
    drift_share_threshold: float | None = None
    n_current: int | None = None
    n_reference: int | None = None
    min_rows: int | None = None
    columns: dict[str, Any] = Field(default_factory=dict)
    skipped_constant_columns: list[str] = Field(default_factory=list)
