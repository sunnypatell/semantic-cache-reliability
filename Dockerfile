# SPDX-License-Identifier: LicenseRef-Proprietary
# Copyright (c) 2026 Sunny Patel. All rights reserved.
# Reproducible environment for the experiments, figures, and tables.
# (The LaTeX build is separate; see build.ps1 / README. This image reproduces the data.)
FROM python:3.12-slim

WORKDIR /app

# CPU-only Torch keeps the image lean; all embeddings run on CPU.
COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt \
 && pip install --no-cache-dir -e .

COPY experiments ./experiments
COPY paper/paper_style.mplstyle ./paper/paper_style.mplstyle

# Datasets and embedding models download on first run to the HF cache.
ENV HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false

# Default: reproduce the whole study (reliability, calibration, downstream, figures, tables).
CMD ["python", "experiments/run_all.py"]
