import numpy as np
import pandas as pd
import pytest

# Le module entier est importé (et non les noms) : waddle.data.validation n'existe
# pas encore, les tests échouent donc un par un faute de code.
from waddle.data import validation

COLUMNS = ["open", "high", "low", "close", "volume"]


def _valid(n: int = 6) -> pd.DataFrame:
    """n barres horaires UTC cohérentes : high >= max(open, close), low <= min(open, close), volume >= 0."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [105.0] * n,
            "low": [95.0] * n,
            "close": [102.0] * n,
            "volume": [10.0] * n,
        },
        index=idx,
    )


def test_valid_frame_has_no_problems():
    assert validation.validate_ohlcv(_valid()) == []


def test_empty_frame_with_right_columns_and_index_is_valid():
    empty = pd.DataFrame(
        {c: pd.Series(dtype=float) for c in COLUMNS},
        index=pd.DatetimeIndex([], tz="UTC"),
    )
    assert validation.validate_ohlcv(empty) == []


# --- missing-columns ---


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_column_is_reported(column):
    df = _valid().drop(columns=[column])
    assert validation.validate_ohlcv(df) == ["missing-columns"]


def test_missing_column_returns_immediately_even_with_other_problems():
    # Index naïf + NaN + colonne absente : seul missing-columns est renvoyé.
    df = _valid().drop(columns=["volume"])
    df.loc[df.index[1], "close"] = np.nan
    df.index = df.index.tz_localize(None)
    assert validation.validate_ohlcv(df) == ["missing-columns"]


# --- bad-index ---


def test_naive_index_is_bad_index():
    df = _valid()
    df.index = df.index.tz_localize(None)
    assert validation.validate_ohlcv(df) == ["bad-index"]


def test_index_not_in_utc_is_bad_index():
    df = _valid()
    df.index = df.index.tz_convert("Europe/Paris")
    assert validation.validate_ohlcv(df) == ["bad-index"]


def test_range_index_is_bad_index():
    df = _valid().reset_index(drop=True)
    assert validation.validate_ohlcv(df) == ["bad-index"]


def test_duplicate_timestamp_is_bad_index():
    idx = pd.DatetimeIndex(
        ["2024-01-01 00:00", "2024-01-01 01:00", "2024-01-01 01:00", "2024-01-01 03:00"],
        tz="UTC",
    )
    df = _valid(4).set_axis(idx)
    assert validation.validate_ohlcv(df) == ["bad-index"]


def test_decreasing_index_is_bad_index():
    assert validation.validate_ohlcv(_valid().iloc[::-1]) == ["bad-index"]


# --- nan-values ---


@pytest.mark.parametrize("column", COLUMNS)
def test_missing_value_is_reported(column):
    df = _valid()
    df.loc[df.index[2], column] = np.nan
    assert validation.validate_ohlcv(df) == ["nan-values"]


# --- high-below-body ---


def test_high_below_body_is_reported():
    # high 101 < max(open, close) = 102, le reste reste cohérent
    df = _valid()
    df.loc[df.index[2], "high"] = 101.0
    assert validation.validate_ohlcv(df) == ["high-below-body"]


# --- low-above-body ---


def test_low_above_body_is_reported():
    # low 101 > min(open, close) = 100, le reste reste cohérent
    df = _valid()
    df.loc[df.index[2], "low"] = 101.0
    assert validation.validate_ohlcv(df) == ["low-above-body"]


# --- high-below-low ---


def test_high_below_low_is_reported():
    # Une ligne avec high < low déclenche forcément aussi une règle de corps
    # (high >= max(open, close) >= min(open, close) >= low sinon). Ici high = max(open, close),
    # donc seule low-above-body s'ajoute : c'est le plus petit cas possible.
    df = _valid()
    df.loc[df.index[2], "high"] = 102.0
    df.loc[df.index[2], "low"] = 103.0
    assert validation.validate_ohlcv(df) == ["low-above-body", "high-below-low"]


# --- negative-volume ---


def test_negative_volume_is_reported():
    df = _valid()
    df.loc[df.index[2], "volume"] = -1.0
    assert validation.validate_ohlcv(df) == ["negative-volume"]


def test_repeated_problem_is_reported_once():
    df = _valid()
    df.loc[df.index[1], "volume"] = -1.0
    df.loc[df.index[3], "volume"] = -2.0
    assert validation.validate_ohlcv(df) == ["negative-volume"]


# --- ordre fixe des codes ---


def test_negative_volume_and_nan_are_reported_in_fixed_order():
    df = _valid()
    df.loc[df.index[1], "volume"] = -1.0
    df.loc[df.index[3], "close"] = np.nan
    assert validation.validate_ohlcv(df) == ["nan-values", "negative-volume"]


def test_bad_index_comes_before_nan_values():
    df = _valid()
    df.index = df.index.tz_localize(None)
    df.loc[df.index[2], "open"] = np.nan
    assert validation.validate_ohlcv(df) == ["bad-index", "nan-values"]


def test_nan_then_body_then_volume_are_reported_in_fixed_order():
    # NaN en open sur la ligne 0 (ignorée par les règles de corps), high trop bas
    # sur la ligne 2, volume négatif sur la ligne 1 : l'ordre de sortie reste fixe.
    df = _valid()
    df.loc[df.index[0], "open"] = np.nan
    df.loc[df.index[1], "volume"] = -1.0
    df.loc[df.index[2], "high"] = 101.0
    assert validation.validate_ohlcv(df) == [
        "nan-values",
        "high-below-body",
        "negative-volume",
    ]
