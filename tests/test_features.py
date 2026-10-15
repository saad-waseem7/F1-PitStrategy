from __future__ import annotations

import pandas as pd
import pytest

from src.config import COMPOUNDS, FEATURE_COLUMNS
from src.features import FeaturePreparer


def _row(**overrides) -> pd.DataFrame:
    base = {c: 1.0 for c in FEATURE_COLUMNS}
    base["compound"] = "MEDIUM"
    base.update(overrides)
    return pd.DataFrame([base])


def test_output_has_fixed_columns_order_and_dtypes():
    shuffled = _row()[list(reversed(FEATURE_COLUMNS))].assign(extra=5)
    out = FeaturePreparer().transform(shuffled)
    assert list(out.columns) == list(FEATURE_COLUMNS)
    assert isinstance(out["compound"].dtype, pd.CategoricalDtype)
    assert list(out["compound"].cat.categories) == list(COMPOUNDS)
    assert (out.drop(columns="compound").dtypes == "float64").all()


def test_compound_is_case_insensitive():
    assert FeaturePreparer().transform(_row(compound="soft"))["compound"].iloc[0] == "SOFT"


def test_unknown_compound_raises():
    with pytest.raises(ValueError, match="Unknown compound"):
        FeaturePreparer().transform(_row(compound="PURPLE"))


def test_missing_column_raises():
    with pytest.raises(ValueError, match="Missing feature columns"):
        FeaturePreparer().transform(_row().drop(columns=["tyre_age"]))


def test_nan_is_preserved_for_lag_features():
    out = FeaturePreparer().transform(_row(pace_decay_slope=float("nan")))
    assert out["pace_decay_slope"].isna().all()


def test_input_frame_is_not_mutated():
    frame = _row()
    FeaturePreparer().transform(frame)
    assert frame["compound"].iloc[0] == "MEDIUM" and not isinstance(frame["compound"].dtype, pd.CategoricalDtype)


@pytest.mark.parametrize("wet", ["INTERMEDIATE", "WET", "wet"])
def test_wet_compounds_are_rejected_as_out_of_scope(wet):
    with pytest.raises(ValueError, match="dry races only"):
        FeaturePreparer().transform(_row(compound=wet))
