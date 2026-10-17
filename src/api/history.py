"""Turn a stint's lap history into the lag features, with the same formulas as preprocessing.

`src.data.preprocess._add_targets_and_features` is the reference. tests/test_api_history.py checks that
this module and that function agree row by row, so a change to one without the other fails the suite.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from src.data.preprocess import trailing_slope

HISTORY_FEATURES: tuple[str, ...] = ("prev_lap_delta", "pace_decay_slope", "prev_gap_to_best", "recent_gap_to_best")


def history_features(
    lap_numbers: Sequence[int],
    lap_durations: Sequence[float],
    current_lap: int,
    fuel_s_per_lap: float,
    decay_window: int,
    gap_window: int,
) -> dict[str, float | None]:
    """Lag features for a lap, from the earlier clean laps of the same stint (oldest first).

    Raises ValueError for lap numbers that are not strictly increasing or not before `current_lap`.
    Features that need more laps than were supplied are None.
    """
    if len(lap_numbers) != len(lap_durations):
        raise ValueError("lap_numbers and lap_durations must have the same length")
    if any(b <= a for a, b in zip(lap_numbers, lap_numbers[1:], strict=False)):
        raise ValueError("previous_laps must be ordered by strictly increasing lap_number")
    if lap_numbers and lap_numbers[-1] >= current_lap:
        raise ValueError("previous_laps must all be earlier than lap_number")

    corrected = np.asarray(lap_durations, dtype=float) + fuel_s_per_lap * np.asarray(lap_numbers, dtype=float)
    out: dict[str, float | None] = dict.fromkeys(HISTORY_FEATURES)
    n = len(corrected)
    if n >= 2:
        out["prev_lap_delta"] = float(corrected[-1] - corrected[-2])
    if n >= decay_window:
        slope = trailing_slope(corrected, decay_window)[-1]
        out["pace_decay_slope"] = None if math.isnan(slope) else float(slope)
    if n >= 1:
        best = float(corrected.min())
        out["prev_gap_to_best"] = float(corrected[-1] - best)
        if n >= gap_window:
            out["recent_gap_to_best"] = float(corrected[-gap_window:].mean() - best)
    return out
