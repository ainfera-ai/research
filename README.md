# Ainfera Research

Open science for **outcome-aware inference routing for AI agents** — the public
research surface for Ainfera Inference.

> Ainfera routes every agent call to the model that will *complete the task*,
> learned from outcomes, neutral across providers. This repo holds the
> **method and the results**. It does **not** hold the operated policy, the real
> labeled outcome corpus, or the production judge — those are the moat and stay
> closed.

## What's here

| Path | Contents |
|---|---|
| `preprint/` | arXiv preprint source — the routing-delta result + methodology |
| `methodology/` | Public-safe specs: `q_empirical`, judge protocol, exploration floor |
| `benchmark/` | Reproducible harness — routing vs baselines on **synthetic/public** data |
| `eval/` | Replay + delta-experiment scripts (operate on synthetic data here) |
| `datasets/` | CC0 synthetic outcome sets for reproducibility |

## What's deliberately NOT here

The real labeled corpus, the operated `q_empirical` weights, judge prompts, and
anti-gaming logic. Those live in private infrastructure. Reproducing our headline
number on **synthetic** data is possible from this repo; reproducing it on **our**
data is not — by design.

## Quickstart

```bash
pip install -r benchmark/requirements.txt
python datasets/generate.py           # writes datasets/synthetic_outcomes.jsonl
python benchmark/run_benchmark.py      # routing vs baselines → delta table
```

## The thesis in one line

A rational agent never leaves a router whose all-in task cost is *always* below
its own baseline — and the gap widens as the router learns. This repo measures
that gap on open data so anyone can check the shape of the claim.

## Related

- `ainfera-ai/specs` — open methodology + API contracts (CC-BY 4.0)
- `ainfera-ai/routing` — policy templates + production methodology
- `ainfera-ai/verify` — offline audit-chain verifier

## License

Code: Apache 2.0 (`LICENSE`). Papers: CC-BY 4.0 (`preprint/LICENSE`).
Datasets: CC0 (`datasets/LICENSE`).
