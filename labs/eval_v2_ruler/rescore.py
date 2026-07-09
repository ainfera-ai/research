#!/usr/bin/env python3
"""rescore.py — re-evaluate existing ruler_scores.json with updated gates.

Re-scores WITHOUT re-calling the gateway. Reads the stored per-task results
and re-applies gate thresholds. Also deduplicates: when the same base model
is served by multiple providers, keeps only the best instance (highest
accuracy, then lowest latency, then lowest tokens).

Usage:
    python3 -m labs.eval_v2_ruler.rescore [input.json] [output_dir]
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Gate thresholds (matching updated config.py)
G1_THRESHOLD = 0.99
G2_THRESHOLD = 0.95
G3_THRESHOLD_MS = 8000.0
G4_SOFT_CEILING = 2000.0

# Known marketplace provider suffixes in catalog slugs
# (premium/direct models have no suffix)
PROVIDER_SUFFIXES = [
    "together", "novita", "deepinfra", "groq", "fireworks",
]


def parse_slug(slug: str) -> tuple[str, str]:
    """Split a model slug into (base_model, provider).

    e.g. "gpt-oss-120b-fireworks" → ("gpt-oss-120b", "fireworks")
         "grok-4"                 → ("grok-4", "direct")
    """
    for p in PROVIDER_SUFFIXES:
        if slug.endswith(f"-{p}"):
            base = slug[: -(len(p) + 1)]
            # Make sure the base isn't empty (edge case)
            if base:
                return base, p
    return slug, "direct"


def reeval_model(r: dict) -> dict:
    """Re-evaluate a single model's gates with updated thresholds."""
    tasks = r.get("task_scores", [])
    n_tasks = r.get("n_tasks", len(tasks))

    # Count successes (g1_pass and g2_pass both true)
    g1_passes = sum(1 for t in tasks if t.get("g1_pass"))
    g2_passes = sum(1 for t in tasks if t.get("g2_pass"))
    n_errors = sum(1 for t in tasks if t.get("error"))
    n_successes = sum(1 for t in tasks if t.get("g1_pass") and t.get("g2_pass"))

    # G1: fraction with valid tool call
    g1_score = g1_passes / n_tasks if n_tasks else 0.0
    g1_pass = g1_score >= G1_THRESHOLD

    # G2: fraction with correct tool+args
    g2_score = g2_passes / n_tasks if n_tasks else 0.0
    g2_pass = g2_score >= G2_THRESHOLD

    # G3: median latency (p50) — using the median of successful task latencies
    latencies = [t.get("latency_ms", 0) for t in tasks if t.get("latency_ms", 0) > 0]
    if latencies:
        latencies.sort()
        mid = len(latencies) // 2
        g3_score = latencies[mid] if len(latencies) % 2 == 1 else (latencies[mid - 1] + latencies[mid]) / 2
    else:
        g3_score = 0
    g3_pass = 0 < g3_score <= G3_THRESHOLD_MS

    # G4: median tokens per successful task
    success_tokens = [t.get("total_tokens", 0) for t in tasks if t.get("g1_pass") and t.get("g2_pass")]
    if success_tokens:
        success_tokens.sort()
        mid = len(success_tokens) // 2
        g4_score = success_tokens[mid] if len(success_tokens) % 2 == 1 else (success_tokens[mid - 1] + success_tokens[mid]) / 2
    else:
        g4_score = -1
    g4_pass = 0 < g4_score <= G4_SOFT_CEILING if g4_score > 0 else False

    overall_pass = g1_pass and g2_pass and g3_pass and g4_pass

    return {
        **r,
        "n_successes": n_successes,
        "g1_score": round(g1_score, 4),
        "g1_pass": g1_pass,
        "g2_score": round(g2_score, 4),
        "g2_pass": g2_pass,
        "g3_score": round(g3_score, 1),
        "g3_pass": g3_pass,
        "g4_score": round(g4_score, 1),
        "g4_pass": g4_pass,
        "overall_pass": overall_pass,
        "n_errors": n_errors,
    }


def deduplicate(results: list[dict]) -> tuple[list[dict], list[tuple[str, str, str]]]:
    """Deduplicate: for each base model, keep only the best provider instance.

    Best = highest n_successes, then lowest g3_score (latency), then lowest g4_score (tokens).
    Models with no provider suffix (premium/direct) are always kept.
    """
    groups: dict[str, list[dict]] = defaultdict(list)

    for r in results:
        base, provider = parse_slug(r["model_slug"])
        r["_base_model"] = base
        r["_provider"] = provider
        groups[base].append(r)

    deduped: list[dict] = []
    removed: list[tuple[str, str, str]] = []  # (base, removed_slug, kept_slug)

    for base, entries in groups.items():
        if len(entries) == 1:
            deduped.append(entries[0])
        else:
            # Sort: highest success first, then lowest latency, then lowest tokens
            ranked = sorted(
                entries,
                key=lambda x: (
                    -x["n_successes"],
                    x["g3_score"] if x["g3_score"] > 0 else 999999,
                    x["g4_score"] if x["g4_score"] > 0 else 999999,
                ),
            )
            best = ranked[0]
            deduped.append(best)
            for r in ranked[1:]:
                removed.append((base, r["model_slug"], best["model_slug"]))

    return deduped, removed


def main():
    input_path = sys.argv[1] if len(sys.argv) > 1 else "labs/eval-v2-results-full/ruler_scores.json"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "labs/eval-v2-results-full"

    with open(input_path) as f:
        data = json.load(f)

    results = data["models"]
    print(f"Loaded {len(results)} models from {input_path}")
    print(f"Old gates: G3 <= {data['gates']['G3_threshold_ms']}ms")
    print(f"New gates: G3 <= {G3_THRESHOLD_MS}ms")
    print()

    # Re-evaluate all models with new gates
    reevaluated = [reeval_model(r) for r in results]

    # Before dedup
    passed_before = sum(1 for r in reevaluated if r["overall_pass"])
    print(f"Before dedup: {len(reevaluated)} models, {passed_before} passing")

    # Deduplicate
    deduped, removed = deduplicate(reevaluated)
    print(f"After dedup:  {len(deduped)} models ({len(removed)} duplicates removed)")
    passed_after = sum(1 for r in deduped if r["overall_pass"])
    print(f"Passing after dedup: {passed_after}")
    print()

    # Show removed duplicates
    if removed:
        print(f"=== REMOVED DUPLICATES ({len(removed)}) ===")
        for base, removed_slug, kept_slug in removed:
            print(f"  {base}: removed {removed_slug}, kept {kept_slug}")
        print()

    # Sort: passing first, then by G3 (latency) ascending
    deduped.sort(key=lambda r: (not r["overall_pass"], r["g3_score"] if r["g3_score"] > 0 else 999999))

    # Print final scorecard
    passing = [r for r in deduped if r["overall_pass"]]
    print(f"=== FINAL SCORECARD — {len(passing)} models PASS (deduplicated, G3 <= {G3_THRESHOLD_MS:.0f}ms) ===")
    print()
    print(f"{'Rank':>4}  {'Model':50s}  {'Latency':>8s}  {'Tokens':>7s}  {'Cost':>8s}  {'Provider':>12s}")
    print("-" * 95)
    for i, r in enumerate(passing, 1):
        print(
            f"{i:>4}  {r['model_slug']:50s}  "
            f"{r['g3_score']:>7.0f}ms  {r['g4_score']:>6.0f}tok  "
            f"${r['total_cost_usd']:.4f}  {r['_provider']:>12s}"
        )

    # Also show near-miss (passed accuracy but failed G3)
    near_miss = [r for r in deduped if r["g1_pass"] and r["g2_pass"] and not r["g3_pass"] and r["n_errors"] == 0]
    if near_miss:
        print()
        print(f"=== NEAR-MISS (perfect accuracy, G3 > {G3_THRESHOLD_MS:.0f}ms): {len(near_miss)} ===")
        for r in sorted(near_miss, key=lambda x: x["g3_score"])[:15]:
            print(f"  {r['model_slug']:50s}  {r['g3_score']:>7.0f}ms  {r['g4_score']:>6.0f}tok  {r['_provider']:>12s}")

    # Write corrected JSON
    output_path = os.path.join(output_dir, "ruler_scores_rescored.json")
    output_data = {
        "taskset_version": data.get("taskset_version"),
        "taskset_hash": data.get("taskset_hash"),
        "n_tasks": data.get("n_tasks", 10),
        "n_models_original": len(results),
        "n_models_deduplicated": len(deduped),
        "n_passed": len(passing),
        "gates": {
            "G1_threshold": G1_THRESHOLD,
            "G2_threshold": G2_THRESHOLD,
            "G3_threshold_ms": G3_THRESHOLD_MS,
            "G4_soft_ceiling": G4_SOFT_CEILING,
        },
        "deduplication": {
            "removed_count": len(removed),
            "removed": [{"base": b, "removed": r, "kept": k} for b, r, k in removed],
        },
        "models": [
            {k: v for k, v in r.items() if not k.startswith("_")} for r in deduped
        ],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print()
    print(f"Corrected JSON written: {output_path}")


if __name__ == "__main__":
    main()
