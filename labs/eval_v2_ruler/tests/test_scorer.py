"""Unit tests for the eval-v2 ruler gate scoring logic.

Tests cover:
  - G1 (tool-call validity): well-formed tool call detection
  - G2 (context coherence): correct tool + args matching
  - G3 (latency p95)
  - G4 (cost efficiency)
  - args_match lenient matching (int coercion, case-insensitive strings)
  - taskset hash verification
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.eval_v2_ruler.gateway import CallResult
from labs.eval_v2_ruler.scorer import ModelScore, TaskScore, score_model, score_task, _args_match
from labs.eval_v2_ruler.taskset import RulerTaskSet, canonical_hash, load_frozen


# ── test fixtures ────────────────────────────────────────────────────────────


def _task(expected_tool="get_weather", expected_args=None) -> dict:
    return {
        "id": "test-001",
        "task_type": "tool_use",
        "prompt": "What is the weather in Jakarta?",
        "tools": [
            {"type": "function", "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
            }}
        ],
        "expected_tool": expected_tool,
        "expected_args": expected_args or {"city": "Jakarta"},
    }


def _result(tool_calls=None, error=None, latency_ms=100, tokens=(10, 20)) -> CallResult:
    return CallResult(
        content="",
        tool_calls=tool_calls or [],
        model_used="test-model",
        latency_ms=latency_ms,
        input_tokens=tokens[0],
        output_tokens=tokens[1],
        finish_reason="tool_calls" if tool_calls else "stop",
        tools_dropped=False,
        error=error,
    )


def _tc(name="get_weather", args=None) -> dict:
    return {"name": name, "arguments": args, "args_raw": json.dumps(args) if args else None}


# ── G1: tool-call validity ──────────────────────────────────────────────────


def test_g1_pass_well_formed_call() -> None:
    """Well-formed tool call with known name + valid JSON args → G1 pass."""
    result = _result(tool_calls=[_tc("get_weather", {"city": "Jakarta"})])
    ts = score_task(_task(), result)
    assert ts.g1_pass is True
    assert ts.tool_call_emitted is True
    assert ts.tool_name_valid is True
    assert ts.args_valid_json is True


def test_g1_fail_no_tool_call() -> None:
    """No tool call emitted → G1 fail."""
    result = _result(tool_calls=[])
    ts = score_task(_task(), result)
    assert ts.g1_pass is False
    assert ts.tool_call_emitted is False


def test_g1_fail_unknown_tool_name() -> None:
    """Tool call with unknown name → G1 fail."""
    result = _result(tool_calls=[_tc("launch_rocket", {"target": "mars"})])
    ts = score_task(_task(), result)
    assert ts.g1_pass is False
    assert ts.tool_name_valid is False


def test_g1_fail_invalid_json_args() -> None:
    """Tool call with unparseable JSON args → G1 fail."""
    result = _result(tool_calls=[{"name": "get_weather", "arguments": None, "args_raw": "{broken"}])
    ts = score_task(_task(), result)
    assert ts.g1_pass is False
    assert ts.args_valid_json is False


def test_g1_fail_on_error() -> None:
    """Gateway error → G1 fail."""
    result = _result(error="HTTP 502")
    ts = score_task(_task(), result)
    assert ts.g1_pass is False
    assert ts.error == "HTTP 502"


# ── G2: context coherence ───────────────────────────────────────────────────


def test_g2_pass_correct_tool_and_args() -> None:
    """Correct tool + matching args → G2 pass."""
    result = _result(tool_calls=[_tc("get_weather", {"city": "Jakarta"})])
    ts = score_task(_task(), result)
    assert ts.g2_pass is True
    assert ts.correct_tool is True
    assert ts.args_match is True


def test_g2_fail_wrong_tool() -> None:
    """Wrong tool name → G2 fail."""
    result = _result(tool_calls=[_tc("search_arxiv", {"keyword": "Jakarta"})])
    ts = score_task(_task(), result)
    assert ts.g2_pass is False
    assert ts.correct_tool is False


def test_g2_fail_wrong_args() -> None:
    """Right tool but wrong args → G2 fail."""
    result = _result(tool_calls=[_tc("get_weather", {"city": "Singapore"})])
    ts = score_task(_task(), result)
    assert ts.g2_pass is False
    assert ts.correct_tool is True
    assert ts.args_match is False


# ── args_match lenient matching ─────────────────────────────────────────────


def test_args_match_int_coercion() -> None:
    """Expected int, got string → should coerce and match."""
    assert _args_match({"n": "10"}, {"n": 10}) is True


def test_args_match_case_insensitive_string() -> None:
    """String values should match case-insensitively."""
    assert _args_match({"city": "jakarta"}, {"city": "Jakarta"}) is True
    assert _args_match({"city": "JAKARTA"}, {"city": "Jakarta"}) is True


def test_args_match_extra_keys_ok() -> None:
    """Extra keys in actual are OK (lenient match)."""
    assert _args_match({"city": "Jakarta", "extra": "foo"}, {"city": "Jakarta"}) is True


def test_args_match_missing_key_fails() -> None:
    """Missing expected key → fail."""
    assert _args_match({"country": "Indonesia"}, {"city": "Jakarta"}) is False


def test_args_match_none_actual() -> None:
    """None actual → fail (unless expected is also None)."""
    assert _args_match(None, {"city": "Jakarta"}) is False
    assert _args_match(None, None) is True


# ── G3: latency p95 ─────────────────────────────────────────────────────────


def test_g3_p95_calculation() -> None:
    """p95 latency should be the 95th percentile of successful call latencies."""
    # 10 tasks with latencies 100-1000ms
    scores = []
    for i in range(10):
        result = _result(
            tool_calls=[_tc("get_weather", {"city": "Jakarta"})],
            latency_ms=100 + i * 100,
        )
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    # p95 of [100,200,...,1000] → 950-1000 range
    assert ms.g3_score >= 900
    assert ms.g3_pass is True  # under 3000ms threshold


def test_g3_fail_high_latency() -> None:
    """p95 above 3000ms → G3 fail."""
    scores = []
    for i in range(10):
        result = _result(
            tool_calls=[_tc("get_weather", {"city": "Jakarta"})],
            latency_ms=5000 + i * 100,
        )
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    assert ms.g3_pass is False
    assert ms.g3_score > 3000


# ── G4: cost efficiency ─────────────────────────────────────────────────────


def test_g4_tokens_per_success() -> None:
    """G4 = total tokens / n_successes."""
    scores = []
    for i in range(10):
        result = _result(
            tool_calls=[_tc("get_weather", {"city": "Jakarta"})],
            tokens=(10, 20),  # 30 tokens per task
        )
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    assert ms.n_successes == 10
    assert ms.g4_score == 30.0  # 30 tokens/task × 10 tasks = 300 total / 10 successes = 30
    assert ms.g4_pass is True  # under 2000 ceiling


def test_g4_fail_no_successes() -> None:
    """Zero successes → G4 = inf → fail."""
    scores = []
    for i in range(10):
        result = _result(tool_calls=[])  # no tool calls → G1+G2 fail
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    assert ms.n_successes == 0
    assert ms.g4_pass is False
    assert ms.g4_score == -1  # inf sentinel


# ── overall pass/fail ───────────────────────────────────────────────────────


def test_overall_pass_all_gates() -> None:
    """All gates pass → overall pass."""
    scores = []
    for i in range(10):
        result = _result(
            tool_calls=[_tc("get_weather", {"city": "Jakarta"})],
            latency_ms=200,
            tokens=(10, 20),
        )
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    assert ms.overall_pass is True
    assert ms.g1_pass and ms.g2_pass and ms.g3_pass and ms.g4_pass


def test_overall_fail_on_g1() -> None:
    """G1 fail → overall fail."""
    scores = []
    for i in range(10):
        result = _result(tool_calls=[])  # G1 fail
        scores.append(score_task(_task(), result))
    ms = score_model("test-model", scores)
    assert ms.overall_pass is False
    assert ms.g1_pass is False


# ── edge cases ──────────────────────────────────────────────────────────────


def test_empty_scores() -> None:
    """No task scores → all zeros, overall fail."""
    ms = score_model("empty-model", [])
    assert ms.n_tasks == 0
    assert ms.overall_pass is False


# ── taskset hash verification ───────────────────────────────────────────────


def test_taskset_load_and_verify() -> None:
    """The frozen task set loads and its hash matches."""
    fixture = Path(__file__).parent.parent / "fixtures" / "tool_use_tasks.json"
    ts = load_frozen(fixture)
    assert ts.version == "eval-v2-ruler-tool-use-v1"
    assert ts.n == 10
    assert len(ts.hash) == 64  # SHA-256


def test_taskset_hash_drift_detected() -> None:
    """A tampered task set should raise TaskSetIntegrityError."""
    import tempfile
    data = {
        "version": "test-v1",
        "hash": "0000000000000000000000000000000000000000000000000000000000000000",
        "tasks": [{"id": "t1", "prompt": "test", "tools": [], "expected_tool": "", "expected_args": {}}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name
    from labs.eval_v2_ruler.taskset import TaskSetIntegrityError
    with pytest.raises(TaskSetIntegrityError):
        load_frozen(path)
    Path(path).unlink()
