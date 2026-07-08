"""scorer.py — G1-G4 gate scoring logic.

Maps raw CallResults into per-model gate scores + pass/fail verdicts.

Gates:
  G1  tool-call validity ≥99%  — well-formed tool call (known name, valid JSON args)
  G2  context coherence ≥95%   — correct tool name + semantically valid args
  G3  latency p95 < 3000ms     — first-tool-call latency
  G4  cost efficiency          — tokens per successful task (soft ceiling 2000)
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from labs.eval_v2_ruler import config
from labs.eval_v2_ruler.gateway import CallResult


@dataclass(frozen=True)
class TaskScore:
    """One task's raw scoring components for one model."""

    task_id: str
    # G1: well-formed tool call?
    tool_call_emitted: bool       # did the model emit any tool call?
    tool_name_valid: bool         # is the tool name in the provided schema?
    args_valid_json: bool         # are the arguments parseable JSON?
    g1_pass: bool                 # well-formed = valid name + valid args

    # G2: correct tool + right args?
    correct_tool: bool            # did it call the expected tool?
    args_match: bool              # do the args match the expected args?
    g2_pass: bool                 # correct tool + args match

    # G3: latency
    latency_ms: float

    # G4: cost
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # error
    error: str | None


@dataclass(frozen=True)
class ModelScore:
    """Aggregated gate scores for one model."""

    model_slug: str
    n_tasks: int
    n_successes: int  # tasks where g1 + g2 both pass

    # G1
    g1_score: float           # fraction of well-formed tool calls
    g1_pass: bool

    # G2
    g2_score: float           # fraction of correct tool + args
    g2_pass: bool

    # G3
    g3_score: float           # p95 latency in ms
    g3_pass: bool

    # G4
    g4_score: float           # tokens per successful task
    g4_pass: bool

    # overall
    overall_pass: bool

    # cost
    total_cost_usd: float
    total_input_tokens: int
    total_output_tokens: int

    # detail
    task_scores: tuple[TaskScore, ...] = field(default_factory=tuple)


def score_task(task: dict[str, Any], result: CallResult) -> TaskScore:
    """Score one task's CallResult against the expected tool + args."""
    task_id = task.get("id", "")
    expected_tool = task.get("expected_tool", "")
    expected_args = task.get("expected_args", {})

    # Get the set of valid tool names from the task's tools[]
    valid_names = set()
    for tool in task.get("tools", []):
        fn = tool.get("function") or tool
        if fn.get("name"):
            valid_names.add(fn["name"])

    # Error case — model didn't respond at all
    if result.error:
        return TaskScore(
            task_id=task_id,
            tool_call_emitted=False, tool_name_valid=False, args_valid_json=False,
            g1_pass=False, correct_tool=False, args_match=False, g2_pass=False,
            latency_ms=result.latency_ms,
            input_tokens=result.input_tokens, output_tokens=result.output_tokens,
            total_tokens=result.input_tokens + result.output_tokens,
            error=result.error,
        )

    # Extract the first tool call (we score on the first call)
    tc = result.tool_calls[0] if result.tool_calls else None
    emitted = tc is not None
    name = tc.get("name") if tc else None
    args = tc.get("arguments") if tc else None

    # G1: well-formed tool call
    name_valid = name in valid_names if name else False
    args_valid = isinstance(args, dict)  # parsed JSON → dict
    g1 = emitted and name_valid and args_valid

    # G2: correct tool + args match
    correct = name == expected_tool if name else False
    args_ok = _args_match(args, expected_args) if isinstance(args, dict) else False
    g2 = correct and args_ok

    return TaskScore(
        task_id=task_id,
        tool_call_emitted=emitted,
        tool_name_valid=name_valid,
        args_valid_json=args_valid,
        g1_pass=g1,
        correct_tool=correct,
        args_match=args_ok,
        g2_pass=g2,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        total_tokens=result.input_tokens + result.output_tokens,
        error=None,
    )


def _args_match(actual: dict | None, expected: dict | None) -> bool:
    """Check if the actual args semantically match the expected args.

    We do a lenient match: the expected key-value pairs must be present in
    the actual args, with matching values (case-insensitive for strings).
    Extra keys in actual are OK (some models add context).
    """
    if actual is None or expected is None:
        return actual == expected
    for key, exp_val in expected.items():
        if key not in actual:
            return False
        act_val = actual[key]
        # Type coercion: expected int, got string
        if isinstance(exp_val, int) and not isinstance(exp_val, bool):
            try:
                act_val = int(act_val)
            except (ValueError, TypeError):
                return False
        elif isinstance(exp_val, str) and isinstance(act_val, str):
            if act_val.lower().strip() != exp_val.lower().strip():
                return False
        elif isinstance(exp_val, str) and isinstance(act_val, (int, float)):
            # numeric string expected but got number — coerce
            if str(act_val).lower().strip() != exp_val.lower().strip():
                return False
        elif act_val != exp_val:
            return False
    return True


def score_model(model_slug: str, task_scores: list[TaskScore], cost_usd: float = 0.0) -> ModelScore:
    """Aggregate per-task scores into per-model gate verdicts."""
    n = len(task_scores)
    if n == 0:
        return ModelScore(
            model_slug=model_slug, n_tasks=0, n_successes=0,
            g1_score=0.0, g1_pass=False,
            g2_score=0.0, g2_pass=False,
            g3_score=0.0, g3_pass=False,
            g4_score=0.0, g4_pass=False,
            overall_pass=False,
            total_cost_usd=0.0, total_input_tokens=0, total_output_tokens=0,
            task_scores=(),
        )

    g1_score = sum(1 for t in task_scores if t.g1_pass) / n
    g2_score = sum(1 for t in task_scores if t.g2_pass) / n
    n_successes = sum(1 for t in task_scores if t.g1_pass and t.g2_pass)

    # G3: p95 latency (only on successful calls — errored calls have latency 0)
    latencies = sorted(t.latency_ms for t in task_scores if t.latency_ms > 0)
    if latencies:
        p95_idx = max(0, int(len(latencies) * 0.95) - 1)
        p95 = latencies[p95_idx]
    else:
        p95 = 0.0

    # G4: tokens per successful task
    total_tokens = sum(t.total_tokens for t in task_scores)
    g4 = total_tokens / n_successes if n_successes > 0 else float("inf")

    g1_pass = g1_score >= config.G1_THRESHOLD
    g2_pass = g2_score >= config.G2_THRESHOLD
    g3_pass = p95 <= config.G3_THRESHOLD_MS
    g4_pass = g4 <= config.G4_SOFT_CEILING and n_successes > 0

    return ModelScore(
        model_slug=model_slug,
        n_tasks=n,
        n_successes=n_successes,
        g1_score=round(g1_score, 4),
        g1_pass=g1_pass,
        g2_score=round(g2_score, 4),
        g2_pass=g2_pass,
        g3_score=round(p95, 1),
        g3_pass=g3_pass,
        g4_score=round(g4, 1) if g4 != float("inf") else -1,
        g4_pass=g4_pass,
        overall_pass=g1_pass and g2_pass and g3_pass and g4_pass,
        total_cost_usd=round(cost_usd, 6),
        total_input_tokens=sum(t.input_tokens for t in task_scores),
        total_output_tokens=sum(t.output_tokens for t in task_scores),
        task_scores=tuple(task_scores),
    )
