"""Fleet paper trading report — equity curves + z-score distribution + PnL per trade.

Usage:
    uv run scripts/plot_fleet_report.py          # display
    uv run scripts/plot_fleet_report.py --save   # save to data/plots/YYYY-MM-DD_fleet_report.png
"""
from __future__ import annotations

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COMBOS = [
    {"label": "combo1  e=2.0/x=0.5", "db": "data/fleet_combo1.db", "color": "#1f77b4"},
    {"label": "combo2  e=2.2/x=1.0", "db": "data/fleet_combo2.db", "color": "#ff7f0e"},
    {"label": "combo3  e=2.5/x=0.5", "db": "data/fleet_combo3.db", "color": "#2ca02c"},
    {"label": "combo4  e=2.7/x=1.0", "db": "data/fleet_combo4.db", "color": "#d62728"},
]


def _load_equity(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT ts, nav FROM bot_equity_snapshots ORDER BY ts", conn
        )
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


def _load_trades(db_path: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            SELECT t.pnl_net, p.entry_z
            FROM bot_trades t
            JOIN bot_positions p ON t.position_id = p.id
            ORDER BY t.exit_ts
            """,
            conn,
        )
    return df


def main(save: bool) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))
    fig.suptitle(
        f"Waddle Fleet — Paper Trading Report  ({datetime.now().strftime('%Y-%m-%d')})",
        fontsize=13,
        fontweight="bold",
    )

    # ── 1. Equity curves ──────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.axhline(1000, color="gray", linestyle="--", linewidth=0.8, label="$1000 baseline")
    for combo in COMBOS:
        try:
            eq = _load_equity(combo["db"])
            ax1.plot(eq["ts"], eq["nav"], label=combo["label"], color=combo["color"])
        except Exception:
            pass
    ax1.set_title("Equity Curve — NAV ($) par combo")
    ax1.set_ylabel("NAV ($)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # ── 2. Z-score distribution at entry ─────────────────────────────────────
    ax2 = axes[1]
    all_entry_z: list[float] = []
    for combo in COMBOS:
        try:
            trades = _load_trades(combo["db"])
            all_entry_z.extend(trades["entry_z"].tolist())
        except Exception:
            pass
    if all_entry_z:
        ax2.hist(all_entry_z, bins=20, edgecolor="black", alpha=0.75, color="#5c85d6")
    ax2.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_title("Distribution des z-scores à l'entrée (toutes combos)")
    ax2.set_xlabel("Z-score d'entrée")
    ax2.set_ylabel("Nombre de trades")
    ax2.grid(True, alpha=0.3)

    # ── 3. PnL per trade ─────────────────────────────────────────────────────
    ax3 = axes[2]
    all_pnl: list[float] = []
    for combo in COMBOS:
        try:
            trades = _load_trades(combo["db"])
            all_pnl.extend(trades["pnl_net"].tolist())
        except Exception:
            pass
    if all_pnl:
        sorted_pnl = sorted(all_pnl)
        colors = ["#2ca02c" if p >= 0 else "#d62728" for p in sorted_pnl]
        ax3.bar(range(len(sorted_pnl)), sorted_pnl, color=colors)
        wins = sum(1 for p in all_pnl if p >= 0)
        ax3.set_title(
            f"PnL net par trade — {wins}/{len(all_pnl)} wins"
            f"  (total: ${sum(all_pnl):+.2f})"
        )
    else:
        ax3.set_title("PnL net par trade — aucune trade fermée")
    ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xlabel("Trade # (trié par PnL)")
    ax3.set_ylabel("PnL ($)")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        out_dir = Path("data/plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{datetime.now().strftime('%Y-%m-%d')}_fleet_report.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true", help="Export PNG instead of display")
    args = parser.parse_args()
    main(args.save)
