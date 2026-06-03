# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Wait for the reliability run to finish, then run the rest of the pipeline.

Intended to be launched in the background while run_reliability.py is still going. It polls
for the 21 completed reports (7 encoders x 3 domains), then runs calibration, the downstream
fault-injection study, and the figure and table generators in sequence. This lets the full
compute chain complete unattended after a single launch.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SUMMARY = ROOT / "results" / "reliability_summary.csv"
EXPECTED_REPORTS = 21  # 6 sentence encoders + 1 lexical control, times 3 domains


def reliability_done() -> bool:
    n_json = len(list((ROOT / "results" / "reliability").glob("*.json")))
    if n_json < EXPECTED_REPORTS:
        return False
    if not SUMMARY.exists():
        return False
    try:
        return len(pd.read_csv(SUMMARY)) >= EXPECTED_REPORTS
    except Exception:
        return False


def main() -> None:
    print("[finalize] waiting for the reliability run to complete ...", flush=True)
    waited = 0
    while not reliability_done():
        time.sleep(20)
        waited += 20
        if waited % 120 == 0:
            n = len(list((ROOT / "results" / "reliability").glob("*.json")))
            print(f"[finalize] {n}/{EXPECTED_REPORTS} reports after {waited}s", flush=True)
    print("[finalize] reliability complete; running downstream stages.", flush=True)

    stages = [
        ["run_calibration.py"],
        ["run_downstream.py", "--n", "400"],
        ["make_figures.py"],
        ["make_tables.py"],
    ]
    for script, *args in stages:
        print(f"\n[finalize] === {script} {' '.join(args)} ===", flush=True)
        r = subprocess.run([sys.executable, str(HERE / script), *args])
        print(f"[finalize] {script} exit={r.returncode}", flush=True)
    print("\n[finalize] all post-reliability stages done.", flush=True)


if __name__ == "__main__":
    main()
