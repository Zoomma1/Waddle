import pandas as pd
import pytest

# Le module entier est importé (et non les noms) : find_gaps n'existe pas encore,
# un import nominatif ferait échouer toute la collecte au lieu d'un échec par test.
from waddle.data import gaps

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _t(hours: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=hours)


def _index(hours):
    """DatetimeIndex UTC à partir d'heures depuis T0."""
    return pd.DatetimeIndex([_t(h) for h in hours])


# --- trous simples (freq par défaut "1h") ---

def test_single_missing_bar_is_reported_as_hole():
    # 02:00 et 03:00 manquent
    holes = gaps.find_gaps(_index([0, 1, 4]))

    assert holes == [(_t(2), _t(3))]


def test_one_missing_bar_gives_degenerate_hole():
    # seul 02:00 manque : premier et dernier manquant sont identiques
    holes = gaps.find_gaps(_index([0, 1, 3]))

    assert holes == [(_t(2), _t(2))]


def test_default_freq_is_one_hour():
    # sans freq explicite, la grille est 1h : 01:00 manque
    holes = gaps.find_gaps(_index([0, 2]))

    assert holes == [(_t(1), _t(1))]


def test_two_holes_are_returned_in_chronological_order():
    holes = gaps.find_gaps(_index([0, 1, 4, 5, 8]), freq="1h")

    assert holes == [(_t(2), _t(3)), (_t(6), _t(7))]


def test_holes_are_returned_as_tz_aware_utc_timestamps():
    holes = gaps.find_gaps(_index([0, 1, 4]))

    start, end = holes[0]
    assert isinstance(start, pd.Timestamp) and start.tzinfo is not None
    assert isinstance(end, pd.Timestamp) and end.tzinfo is not None
    assert str(start.tz) == "UTC" and str(end.tz) == "UTC"


# --- séries sans trou ---

def test_contiguous_series_has_no_hole():
    assert gaps.find_gaps(_index([0, 1, 2, 3, 4])) == []


def test_empty_index_has_no_hole():
    assert gaps.find_gaps(pd.DatetimeIndex([], tz="UTC")) == []


def test_single_element_index_has_no_hole():
    assert gaps.find_gaps(_index([0])) == []


# --- freq="8h" (grille des funding rates) ---

def test_eight_hour_grid_without_gap_has_no_hole():
    # 00:00, 08:00, 16:00
    assert gaps.find_gaps(_index([0, 8, 16]), freq="8h") == []


def test_eight_hour_grid_reports_missing_settlements():
    # 00:00, 08:00, puis 40:00 : 16:00, 24:00 et 32:00 manquent
    holes = gaps.find_gaps(_index([0, 8, 40]), freq="8h")

    assert holes == [(_t(16), _t(32))]


# --- erreurs de validation ---

def test_naive_index_raises_value_error():
    naive = _index([0, 1, 4]).tz_localize(None)

    with pytest.raises(ValueError):
        gaps.find_gaps(naive)


def test_duplicate_timestamp_raises_value_error():
    with pytest.raises(ValueError):
        gaps.find_gaps(_index([0, 1, 1, 2]))


def test_out_of_order_index_raises_value_error():
    with pytest.raises(ValueError):
        gaps.find_gaps(_index([0, 2, 1]))


def test_unaligned_timestamp_raises_value_error():
    # 01:30 n'est pas sur la grille 1h
    idx = pd.DatetimeIndex([_t(0), _t(1.5), _t(2)])

    with pytest.raises(ValueError):
        gaps.find_gaps(idx, freq="1h")


def test_timestamp_off_eight_hour_grid_raises_value_error():
    # 04:00 n'est pas sur la grille 8h
    with pytest.raises(ValueError):
        gaps.find_gaps(_index([0, 4, 8]), freq="8h")
