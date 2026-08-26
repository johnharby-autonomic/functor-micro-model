#!/bin/bash

python3 -m sweep_prob \
 --windows 8 16 32 64 \
 --bandwidths 0.25 0.5 1.0 \
 --updates 400 --predictions-per-update 10 \
 --output-json sweep_results.json

