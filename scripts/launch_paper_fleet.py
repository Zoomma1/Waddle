"""Launch the Waddle paper trading fleet -- 4 bot instances with different params.

Each instance runs as a subprocess of `scripts/run_bot.py` with its own config + params.
All processes are monitored; Ctrl+C kills all of them gracefully.

Usage:
    uv run scripts/launch_paper_fleet.py
    uv run scripts/launch_paper_fleet.py --combos 1,2  (run only combo 1 and 2)
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

FLEET_COMBOS = {
    1: {"label": "entry=2.0/exit=0.5 (most active)", "config": "config/fleet/combo1_bot.yaml", "params": "config/fleet/combo1_params.json", "log": "logs/fleet_combo1.log"},
    2: {"label": "entry=2.2/exit=1.0 (compromise)", "config": "config/fleet/combo2_bot.yaml", "params": "config/fleet/combo2_params.json", "log": "logs/fleet_combo2.log"},
    3: {"label": "entry=2.5/exit=0.5 (walk-forward stable)", "config": "config/fleet/combo3_bot.yaml", "params": "config/fleet/combo3_params.json", "log": "logs/fleet_combo3.log"},
    4: {"label": "entry=2.7/exit=1.0 (conservative)", "config": "config/fleet/combo4_bot.yaml", "params": "config/fleet/combo4_params.json", "log": "logs/fleet_combo4.log"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Waddle paper fleet")
    parser.add_argument(
        "--combos",
        default=None,
        help="Comma-separated combo IDs to run (default: all). Example: --combos 1,3",
    )
    args = parser.parse_args()

    if args.combos:
        selected = [int(c.strip()) for c in args.combos.split(",")]
        for c in selected:
            if c not in FLEET_COMBOS:
                raise SystemExit(f"ERROR: unknown combo {c}, valid: {sorted(FLEET_COMBOS)}")
    else:
        selected = sorted(FLEET_COMBOS.keys())

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  WADDLE PAPER FLEET")
    print("=" * 60)
    print(f"  Launching {len(selected)} instance(s)")
    print()

    processes: dict[int, tuple[subprocess.Popen, Path]] = {}

    for combo_id in selected:
        combo = FLEET_COMBOS[combo_id]
        log_path = Path(combo["log"])

        log_file = open(log_path, "w", encoding="utf-8")

        cmd = [
            sys.executable, "-u",
            "scripts/run_bot.py",
            "--config", combo["config"],
            "--active-params", combo["params"],
        ]

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}

        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
        )

        processes[combo_id] = (proc, log_path)
        print(f"  [Combo {combo_id}] PID {proc.pid} -- {combo['label']}")
        print(f"             config: {combo['config']}")
        print(f"             log:    {combo['log']}")
        print()

    print("=" * 60)
    print(f"  Fleet started: {len(processes)} instance(s)")
    print("  Press Ctrl+C to stop all instances")
    print("=" * 60)
    print()

    try:
        while True:
            any_alive = False
            for combo_id, (proc, log_path) in list(processes.items()):
                ret = proc.poll()
                if ret is not None:
                    label = FLEET_COMBOS[combo_id]["label"]
                    if ret == 0:
                        print(f"  [Combo {combo_id}] exited cleanly (code 0)")
                    else:
                        print(f"  WARNING: [Combo {combo_id}] crashed (exit code {ret}) -- {label}")
                        print(f"           Check {log_path} for details")
                    del processes[combo_id]
                else:
                    any_alive = True

            if not any_alive and not processes:
                print("  All fleet instances have exited.")
                break

            time.sleep(2)

    except KeyboardInterrupt:
        print("\n  Ctrl+C received -- shutting down fleet...")
        for combo_id, (proc, _) in processes.items():
            print(f"  [Combo {combo_id}] sending terminate (PID {proc.pid})")
            proc.terminate()

        deadline = time.time() + 10
        for combo_id, (proc, _) in processes.items():
            remaining = max(0, deadline - time.time())
            try:
                proc.wait(timeout=remaining)
                print(f"  [Combo {combo_id}] stopped")
            except subprocess.TimeoutExpired:
                print(f"  [Combo {combo_id}] force killing (PID {proc.pid})")
                proc.kill()
                proc.wait()

        print("  Fleet shutdown complete.")


if __name__ == "__main__":
    main()
