import pandas as pd

# Le module entier est importé (et non les noms) : count_funding_rates n'existe
# pas encore, un import nominatif ferait échouer toute la collecte au lieu d'un
# échec par test.
from waddle.data import storage

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
SYMBOL = "BTC/USDT:USDT"
OTHER = "ETH/USDT:USDT"


def _funding(n, first=0, rate=0.0001):
    """n settlements de funding, à 8h d'intervalle, à partir du settlement `first`."""
    idx = pd.date_range(T0 + pd.Timedelta(hours=8 * first), periods=n, freq="8h", tz="UTC")
    return pd.DataFrame({"funding_rate": [rate] * n}, index=idx)


def _bars(n, first=0):
    """n barres horaires OHLCV, à partir de l'heure `first`."""
    idx = pd.date_range(T0 + pd.Timedelta(hours=first), periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0] * n,
            "high": [11.0] * n,
            "low": [9.0] * n,
            "close": [10.5] * n,
            "volume": [100.0] * n,
        },
        index=idx,
    )


def _count(db, symbol):
    return storage.count_funding_rates(symbol, db_path=db)


def test_count_is_zero_for_unknown_symbol_on_empty_db(tmp_path):
    db = tmp_path / "waddle.db"

    assert _count(db, SYMBOL) == 0


def test_count_is_zero_for_unknown_symbol_on_non_empty_db(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_funding_rates(_funding(3), SYMBOL, db_path=db)

    assert _count(db, OTHER) == 0


def test_count_returns_number_of_saved_settlements(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_funding_rates(_funding(5), SYMBOL, db_path=db)

    assert _count(db, SYMBOL) == 5


def test_rewriting_same_timestamps_does_not_change_count(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_funding_rates(_funding(4, rate=0.0001), SYMBOL, db_path=db)
    # Même (symbol, timestamp), taux différent : upsert, pas d'ajout.
    storage.save_funding_rates(_funding(4, rate=0.0002), SYMBOL, db_path=db)

    assert _count(db, SYMBOL) == 4


def test_overlapping_save_counts_union_of_timestamps(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_funding_rates(_funding(4, first=0), SYMBOL, db_path=db)  # settlements 0..3
    storage.save_funding_rates(_funding(4, first=2), SYMBOL, db_path=db)  # settlements 2..5

    assert _count(db, SYMBOL) == 6


def test_each_symbol_is_counted_separately(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_funding_rates(_funding(3), SYMBOL, db_path=db)
    storage.save_funding_rates(_funding(5), OTHER, db_path=db)

    assert _count(db, SYMBOL) == 3
    assert _count(db, OTHER) == 5


def test_ohlcv_bars_of_same_symbol_are_not_counted(tmp_path):
    db = tmp_path / "waddle.db"
    storage.save_ohlcv(_bars(4), SYMBOL, db_path=db)
    storage.save_ohlcv_bybit(_bars(4), SYMBOL, db_path=db)

    assert _count(db, SYMBOL) == 0

    storage.save_funding_rates(_funding(2), SYMBOL, db_path=db)

    assert _count(db, SYMBOL) == 2


def test_parent_directory_of_db_path_is_created(tmp_path):
    parent = tmp_path / "data" / "nested"
    db = parent / "waddle.db"
    assert not parent.exists()

    assert _count(db, SYMBOL) == 0
    assert parent.is_dir()
