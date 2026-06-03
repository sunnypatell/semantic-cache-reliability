# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
"""Reproduce the entire study end to end.

Runs, in order: the reliability experiment (false-hit economics across encoders and
domains), the calibration comparison, the downstream fault-injection study, and then
renders every figure and table the paper uses. Each stage writes to results/ and the
paper's figures/ and tables/ directories.

    python experiments/run_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

STAGES = [
    ["run_reliability.py", "--max-pairs", "2500", "--n-boot", "1500"],
    ["run_calibration.py"],
    ["run_downstream.py", "--n", "400"],
    ["make_figures.py"],
    ["make_tables.py"],
]


def main() -> None:
    for stage in STAGES:
        script, *args = stage
        print(f"\n{'=' * 70}\n[run_all] {script} {' '.join(args)}\n{'=' * 70}")
        subprocess.run([PY, str(HERE / script), *args], check=True)
    print("\n[run_all] complete. Compile the paper with build.ps1 (or latexmk in paper/).")


if __name__ == "__main__":
    main()
