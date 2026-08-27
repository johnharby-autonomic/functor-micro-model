#!/usr/bin/env bash

set -euo pipefail

python3 -m sweep_prob \
  --routes dist sample mean var stddev \
  --strategies lazy eager \
  --updates 400 --predictions-per-update 10 \
  --trials 3 --warmup 1 \
  --lifecycle \
  --output-json sweep_results.json
