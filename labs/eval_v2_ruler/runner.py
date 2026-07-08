"""runner.py — orchestrate the full ruler run.

For each model in the catalog:
  1. Call each task in the frozen task set with tools[]
  2. Score each CallResult (G1/G2 per task)
  3. Aggregate into per-model gate verdicts
  4. Write the CSV output

Cost guardrail: stops if cumulative spend exceeds COST_CAP_USD.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labs.eval_v2_ruler import config
from labs.eval_v2_ruler.catalog import CatalogModel, fetch_catalog
from labs.eval_v2_ruler.gateway import GatewayClient
from labs.eval_v2_ruler.scorer import ModelScore, TaskScore, score_model, score_task
from labs.eval_v2_ruler.taskset import RulerTaskSet


@dataclass
class RunResult:
    n_models: int
    n_scored: int
    n_passed: int
    total_cost_usd: float
    csv_path: str
    json_path: str
    errors: list[str]


def run_ruler(
    taskset: RulerTaskSet,
    *,
    models: list[CatalogModel] | None = None,
    max_models: int | None = None,
    output_dir: str | None = None,
) -> RunResult:
    """Run the full ruler. Returns a RunResult with paths to artifacts."""
    gw = GatewayClient()
    if models is None:
        models = fetch_catalog()
    if max_models is not None:
        models = models[:max_models]

    output_dir = output_dir or config.OUTPUT_DIR
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    all_scores: list[ModelScore] = []
    cumulative_cost = 0.0
    errors: list[str] = []
    n_passed = 0

    print(f"ruler: scoring {len(models)} models × {taskset.n} tasks "
          f"(cost cap ${config.COST_CAP_USD})", file=sys.stderr)

    for i, model in enumerate(models):
        if cumulative_cost > config.COST_CAP_USD:
            print(f"ruler: COST CAP ${config.COST_CAP_USD} exceeded "
                  f"(spent ${cumulative_cost:.2f}) — stopping", file=sys.stderr)
            break

        slug = model.slug
        task_scores: list[TaskScore] = []
        model_cost = 0.0

        for task in taskset.tasks:
            messages = [{"role": "user", "content": task["prompt"]}]
            tools = task.get("tools")
            result = gw.call(
                model=slug,
                messages=messages,
                tools=tools,
                max_tokens=config.MAX_TOKENS,
            )
            ts = score_task(task, result)
            task_scores.append(ts)

            # estimate cost
            if model.input_cost_per_million and model.output_cost_per_million:
                model_cost += (
                    result.input_tokens * model.input_cost_per_million / 1_000_000
                    + result.output_tokens * model.output_cost_per_million / 1_000_000
                )

            # G3: brief sleep to avoid rate limits
            time.sleep(0.1)

        cumulative_cost += model_cost
        ms = score_model(slug, task_scores, cost_usd=model_cost)
        all_scores.append(ms)
        if ms.overall_pass:
            n_passed += 1

        status = "PASS" if ms.overall_pass else "FAIL"
        print(f"  [{i+1}/{len(models)}] {slug:40s} {status}  "
              f"G1={ms.g1_score:.2f} G2={ms.g2_score:.2f} "
              f"G3={ms.g3_score:.0f}ms G4={ms.g4_score:.0f}tok  "
              f"${model_cost:.4f}  cumul=${cumulative_cost:.2f}",
              file=sys.stderr)

        if ms.n_successes == 0 and all(t.error for t in task_scores):
            errors.append(f"{slug}: all tasks errored")

    # Sort: passing models first, then by G2 score descending
    all_scores.sort(key=lambda s: (not s.overall_pass, -s.g2_score))

    # Write CSV
    csv_path = os.path.join(output_dir, "ruler_scores.csv")
    _write_csv(csv_path, all_scores)

    # Write JSON (full detail)
    json_path = os.path.join(output_dir, "ruler_scores.json")
    _write_json(json_path, all_scores, taskset)

    return RunResult(
        n_models=len(models),
        n_scored=len(all_scores),
        n_passed=n_passed,
        total_cost_usd=round(cumulative_cost, 4),
        csv_path=csv_path,
        json_path=json_path,
        errors=errors,
    )


def _write_csv(path: str, scores: list[ModelScore]) -> None:
    """Write the CSV in the format specified by the task:
    model_id, g1_pass, g1_score, g2_pass, g2_score, g3_pass, g3_score,
    g4_pass, g4_score, overall_pass"""
    fields = [
        "model_id", "g1_pass", "g1_score", "g2_pass", "g2_score",
        "g3_pass", "g3_score", "g4_pass", "g4_score", "overall_pass",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in scores:
            w.writerow({
                "model_id": s.model_slug,
                "g1_pass": s.g1_pass,
                "g1_score": s.g1_score,
                "g2_pass": s.g2_pass,
                "g2_score": s.g2_score,
                "g3_pass": s.g3_pass,
                "g3_score": s.g3_score,
                "g4_pass": s.g4_pass,
                "g4_score": s.g4_score,
                "overall_pass": s.overall_pass,
            })


def _write_json(path: str, scores: list[ModelScore], taskset: RulerTaskSet) -> None:
    """Write full detail JSON for audit."""
    data = {
        "taskset_version": taskset.version,
        "taskset_hash": taskset.hash,
        "n_tasks": taskset.n,
        "n_models": len(scores),
        "n_passed": sum(1 for s in scores if s.overall_pass),
        "gates": {
            "G1_threshold": config.G1_THRESHOLD,
            "G2_threshold": config.G2_THRESHOLD,
            "G3_threshold_ms": config.G3_THRESHOLD_MS,
            "G4_soft_ceiling": config.G4_SOFT_CEILING,
        },
        "models": [
            {
                "model_slug": s.model_slug,
                "n_tasks": s.n_tasks,
                "n_successes": s.n_successes,
                "g1_score": s.g1_score, "g1_pass": s.g1_pass,
                "g2_score": s.g2_score, "g2_pass": s.g2_pass,
                "g3_score": s.g3_score, "g3_pass": s.g3_pass,
                "g4_score": s.g4_score, "g4_pass": s.g4_pass,
                "overall_pass": s.overall_pass,
                "total_cost_usd": s.total_cost_usd,
                "total_input_tokens": s.total_input_tokens,
                "total_output_tokens": s.total_output_tokens,
                "task_scores": [
                    {
                        "task_id": t.task_id,
                        "g1_pass": t.g1_pass,
                        "g2_pass": t.g2_pass,
                        "latency_ms": t.latency_ms,
                        "total_tokens": t.total_tokens,
                        "error": t.error,
                    }
                    for t in s.task_scores
                ],
            }
            for s in scores
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
