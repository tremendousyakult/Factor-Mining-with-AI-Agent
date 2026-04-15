#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. python -m agentic_factor.cli run \
  --config configs/demo_small.yaml \
  --data sample_panel.csv \
  --out rerun_sample \
  --report
