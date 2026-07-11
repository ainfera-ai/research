# Ainfera Research — Open Science for Inference Routing

Open science for outcome-aware inference routing for AI agents. Public research surface for [Ainfera](https://ainfera.ai).

[![CI](https://github.com/ainfera-ai/research/actions/workflows/ci.yml/badge.svg)](https://github.com/ainfera-ai/research/actions)
[![License](https://img.shields.io/badge/license-CC--BY--4.0-green)](LICENSE)

## Overview

This repository contains the research artifacts behind Ainfera's outcome-aware routing methodology. Everything is reproducible — benchmarks, datasets, evaluation harness, and methodology documentation.

## What's Here

| Directory | Description |
|---|---|
| [`benchmark/`](benchmark) | Benchmark harness — `run_benchmark.py` |
| [`methodology/`](methodology) | Methodology docs — judge protocol, empirical Q, exploration floor |
| [`datasets/`](datasets) | Dataset generation + licensing |
| [`eval/`](eval) | Delta evaluation, replay, comparison |
| [`labs/`](labs) | Lab experiments |
| [`preprint/`](preprint) | Preprint manuscripts (LaTeX) |

## Methodology

Ainfera's routing research is built on three principles:

1. **Outcome-aware** — routing decisions learn from observed outcomes (latency, quality, cost), not static model metadata
2. **Reproducible** — every benchmark run is deterministic with pinned seeds and recorded env state
3. **Open** — methodology, datasets, and evaluation code are public under CC-BY 4.0

### Key Documents

- [`methodology/judge-protocol.md`](methodology/judge-protocol.md) — LLM-as-judge evaluation protocol
- [`methodology/q-empirical.md`](methodology/q-empirical.md) — Empirical quality scoring
- [`methodology/exploration-floor.md`](methodology/exploration-floor.md) — Exploration vs. exploitation in routing

## Quick Start

```bash
# Install dependencies
pip install -r benchmark/requirements.txt

# Run the benchmark suite
python benchmark/run_benchmark.py

# Evaluate routing delta (before vs after)
python eval/delta.py --baseline results-baseline.json --candidate results-candidate.json
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

Creative Commons Attribution 4.0 International — see [LICENSE](LICENSE).
