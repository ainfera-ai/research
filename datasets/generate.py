#!/usr/bin/env python3
"""Generate a CC0 SYNTHETIC outcome dataset for the routing benchmark.

Nothing here is real Ainfera data. Latent qualities, prices, and the request
stream are invented so the benchmark is reproducible without touching the
operated corpus. Deterministic given --seed.
"""
import argparse
import json
import os
import random

# 5 synthetic models. Names are generic; the shape (a strong-expensive model, a
# cheap-weak model, and task-specialists) mirrors a real routable set.
MODELS = [
    {"id": "model-a", "price_per_1k": 15.0},  # strong, expensive generalist
    {"id": "model-b", "price_per_1k": 5.0},   # mid all-rounder
    {"id": "model-c", "price_per_1k": 3.0},   # cheap, code-specialist
    {"id": "model-d", "price_per_1k": 2.0},   # cheap, weak generalist
    {"id": "model-e", "price_per_1k": 8.0},   # reasoning-specialist
]
TASKS = ["code", "reasoning", "extraction", "chat", "summarize"]

# Latent true quality (1-5) per (model, task). Specialists beat the expensive
# generalist on their task at a fraction of the price -- that's where routing wins.
LATENT = {
    "model-a": {"code": 4.6, "reasoning": 4.7, "extraction": 4.4, "chat": 4.5, "summarize": 4.5},
    "model-b": {"code": 4.0, "reasoning": 4.0, "extraction": 4.2, "chat": 4.3, "summarize": 4.2},
    "model-c": {"code": 4.7, "reasoning": 3.4, "extraction": 4.0, "chat": 3.6, "summarize": 3.7},
    "model-d": {"code": 3.0, "reasoning": 2.8, "extraction": 3.6, "chat": 4.0, "summarize": 3.9},
    "model-e": {"code": 4.1, "reasoning": 4.8, "extraction": 4.1, "chat": 4.0, "summarize": 4.1},
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of requests")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", default=os.path.dirname(__file__) or ".")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    catalog = {"models": MODELS, "tasks": TASKS, "latent_quality": LATENT}
    with open(os.path.join(args.outdir, "catalog.json"), "w") as f:
        json.dump(catalog, f, indent=2)

    path = os.path.join(args.outdir, "synthetic_outcomes.jsonl")
    with open(path, "w") as f:
        for i in range(args.n):
            task = rng.choice(TASKS)
            tokens = rng.randint(1, 8)          # in thousands
            min_quality = rng.choice([3.5, 4.0, 4.0, 4.2])
            f.write(json.dumps({
                "id": i, "task_type": task,
                "tokens_k": tokens, "min_quality": min_quality,
            }) + "\n")
    print(f"wrote {args.n} requests -> {path}")
    print(f"wrote catalog -> {os.path.join(args.outdir, 'catalog.json')}")

if __name__ == "__main__":
    main()
