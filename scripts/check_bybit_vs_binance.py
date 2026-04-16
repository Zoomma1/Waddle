"""Sanity-check Bybit perpetual vs Binance spot prices for SOL and AVAX.

If the SOL/AVAX backtest edge lives on Bybit, we need to be sure that the
price series used in the original Binance backtest is not radically
different from the actual deployment data. Two questions:

1. **Basis** — how far apart are the spot and perp closes at any given hour?
   For a well-functioning major perpetual with regular funding, this should
   be within a fraction of a percent most of the time. A basis that
   routinely exceeds 0.5% would indicate a real pricing wedge between the
   two markets — and would make the Binance backtest a questionable proxy
   for Bybit deployment.

2. **Log-return correlation** — do the two series move together? A
   correlation below 0.99 on majors would flag either a liquidity problem
   on one side or a genuine structural divergence. We want >=0.99 here;
   anything materially lower is a red flag.

The check runs on the overlapping period only (intersection of Binance and
Bybit covered hours) and reports the stats per symbol.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from waddle.data.storage import load_ohlcv, load_ohlcv_bybit

SPOT_VS_PERP = [
    ("SOL/USDT", "SOL/USDT:USDT"),
    ("AVAX/USDT", "AVAX/USDT:USDT"),
    ("BTC/USDT", "BTC/USDT:USDT"),
    ("ETH/USDT", "ETH/USDT:USDT"),
]

BASIS_FLAG_THRESHOLD = 0.005  # 0.5% absolute
CORR_FLAG_THRESHOLD = 0.99


def check_one(spot_symbol: str, perp_symbol: str) -> dict:
    spot = load_ohlcv(spot_symbol)["close"]
    perp = load_ohlcv_bybit(perp_symbol)["close"]

    if len(spot) == 0 or len(perp) == 0:
        return {
            "spot": spot_symbol,
            "perp": perp_symbol,
            "error": "missing data",
            "n_overlap": 0,
        }

    overlap_start = max(spot.index[0], perp.index[0])
    overlap_end = min(spot.index[-1], perp.index[-1])
    spot_sl = spot[(spot.index >= overlap_start) & (spot.index <= overlap_end)]
    perp_sl = perp[(perp.index >= overlap_start) & (perp.index <= overlap_end)]

    aligned = pd.DataFrame({"spot": spot_sl, "perp": perp_sl}).dropna()

    if len(aligned) == 0:
        return {
            "spot": spot_symbol,
            "perp": perp_symbol,
            "error": "no aligned bars",
            "n_overlap": 0,
        }

    basis = aligned["perp"] / aligned["spot"] - 1.0
    log_ret_spot = np.log(aligned["spot"]).diff().dropna()
    log_ret_perp = np.log(aligned["perp"]).diff().dropna()
    joined_ret = pd.DataFrame({"s": log_ret_spot, "p": log_ret_perp}).dropna()
    corr = float(joined_ret["s"].corr(joined_ret["p"])) if len(joined_ret) > 1 else float("nan")

    flag_basis = (basis.abs() > BASIS_FLAG_THRESHOLD).sum()
    pct_flag = 100 * flag_basis / len(basis)

    return {
        "spot": spot_symbol,
        "perp": perp_symbol,
        "n_overlap": len(aligned),
        "overlap_start": aligned.index[0],
        "overlap_end": aligned.index[-1],
        "mean_basis": float(basis.mean()),
        "std_basis": float(basis.std()),
        "median_basis": float(basis.median()),
        "min_basis": float(basis.min()),
        "max_basis": float(basis.max()),
        "max_abs_basis": float(basis.abs().max()),
        "pct_time_over_50bp": pct_flag,
        "log_return_corr": corr,
        "flagged_corr": corr < CORR_FLAG_THRESHOLD,
    }


def print_one(res: dict) -> None:
    print(f"\n=> {res['spot']}  vs  {res['perp']}")
    if "error" in res:
        print(f"   ERROR: {res['error']}")
        return
    print(f"   overlap:           {res['overlap_start']}  ->  {res['overlap_end']}")
    print(f"   n aligned hours:   {res['n_overlap']}")
    print(f"   basis (perp/spot - 1):")
    print(f"      mean          = {res['mean_basis'] * 100:+.4f}%")
    print(f"      median        = {res['median_basis'] * 100:+.4f}%")
    print(f"      std           = {res['std_basis'] * 100:.4f}%")
    print(f"      min           = {res['min_basis'] * 100:+.4f}%")
    print(f"      max           = {res['max_basis'] * 100:+.4f}%")
    print(f"      max |basis|   = {res['max_abs_basis'] * 100:.4f}%")
    print(f"      time |basis|>50bp = {res['pct_time_over_50bp']:.2f}% of hours")
    flag = " ** FLAGGED **" if res["flagged_corr"] else ""
    print(f"   log-return corr   = {res['log_return_corr']:.6f}{flag}")


def main() -> None:
    print("=" * 92)
    print("WADDLE — Bybit perpetual vs Binance spot consistency check")
    print("=" * 92)
    print(f"Basis flag threshold:       |basis| > {BASIS_FLAG_THRESHOLD * 100:.2f}%")
    print(f"Correlation flag threshold: < {CORR_FLAG_THRESHOLD}")

    results = []
    for spot_sym, perp_sym in SPOT_VS_PERP:
        res = check_one(spot_sym, perp_sym)
        print_one(res)
        results.append(res)

    print("\n" + "=" * 92)
    print("SUMMARY TABLE")
    print("=" * 92)
    header = (
        f"{'symbol':<12} "
        f"{'n_hours':>8} "
        f"{'mean_bp':>10} "
        f"{'std_bp':>10} "
        f"{'max|bp|':>10} "
        f"{'>50bp%':>8} "
        f"{'log_corr':>10}"
    )
    print(header)
    print("-" * len(header))
    any_flag = False
    for r in results:
        if "error" in r:
            print(f"{r['spot']:<12} ERROR: {r['error']}")
            continue
        flag = "  <<" if r["flagged_corr"] else ""
        any_flag = any_flag or r["flagged_corr"]
        print(
            f"{r['spot'].split('/')[0]:<12} "
            f"{r['n_overlap']:>8} "
            f"{r['mean_basis'] * 10000:>+9.2f}  "
            f"{r['std_basis'] * 10000:>9.2f}  "
            f"{r['max_abs_basis'] * 10000:>9.2f}  "
            f"{r['pct_time_over_50bp']:>7.2f}% "
            f"{r['log_return_corr']:>10.6f}{flag}"
        )

    print("\nColumns: basis values are in basis points (1bp = 0.01% = 0.0001).")
    if any_flag:
        print("\n!! WARNING: at least one symbol has log-return correlation < 0.99.")
        print("   This means spot and perp series are materially decoupled.")
        print("   The Binance backtest CANNOT be trusted as a proxy for Bybit deployment.")
    else:
        print("\nAll correlations >= 0.99 — spot and perp move together as expected.")


if __name__ == "__main__":
    main()
