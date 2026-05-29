#!/usr/bin/env bash
# AIN-288 · one-command reproducible delta harness (synthetic dry-run).
#
#   ./eval/run_delta.sh [SEED]      # default seed 7 (CRN — deterministic)
#
# Generates the CC0 synthetic corpus then computes the routed-vs-baselines
# delta table (routed/ainfera_learned vs single_best / cheapest / round_robin
# (random) / agent_baseline (agent-default) / ainfera_static / oracle) and
# writes preprint/results.md. Until a real judged corpus exists this is a
# DRY-RUN (labeled synthetic, not a published claim).
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-7}"
python datasets/generate.py --seed "$SEED"
python eval/delta.py
