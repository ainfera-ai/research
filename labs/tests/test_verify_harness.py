"""AIN-542 Step 1 · Tier A verifier harness unit tests.

Covers each verifier family, the dispatcher, the sizing histogram/SQL, and the
load-bearing substantive-not-liveness invariant.
"""

from __future__ import annotations

import json

from labs.task_verifiability import CANONICAL_TASK_TYPES
from labs.verify_harness import (
    VERIFIABILITY_SIZING_SQL,
    VerifySample,
    verifiability_histogram,
    verify,
    verify_rows,
)


def _openai(text: str = "", tool_calls=None) -> dict:
    msg: dict = {"content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def _anthropic_text(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


# ── code (intrinsic) ─────────────────────────────────────────────────────────


def test_code_valid_python_scores_1() -> None:
    resp = _openai("Here:\n```python\ndef f(x):\n    return x + 1\n```")
    r = verify(VerifySample("code", response_payload=resp))
    assert r.reward == 1.0 and r.reward_source == "verify" and r.mode == "intrinsic"


def test_code_syntax_error_scores_0() -> None:
    resp = _openai("```python\ndef f(:\n    return\n```")
    r = verify(VerifySample("code", response_payload=resp))
    assert r.reward == 0.0 and r.verifiable is True


def test_code_no_block_defers() -> None:
    resp = _openai("I'd recommend refactoring the loop, but need more detail.")
    r = verify(VerifySample("code", response_payload=resp))
    assert r.reward is None and r.reward_source == "" and r.verifiable is False


# ── extraction / schema (intrinsic) ──────────────────────────────────────────


def _json_req() -> dict:
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "required": ["name", "age"]}},
        }
    }


# ── AIN-547 regression set: correctness over format ──────────────────────────


def test_correct_prose_when_json_not_demanded_scores_above_zero() -> None:
    # the κ-fix core: a correct answer must NOT be penalised for being prose when
    # JSON was never demanded (needs gold to judge correctness).
    resp = _anthropic_text("The name is Ada and she is 36.")
    r = verify(VerifySample("extraction", response_payload=resp, expected="Ada"))
    assert r.reward is not None and r.reward > 0


def test_structured_demanded_valid_json_scores_1() -> None:
    resp = _anthropic_text(json.dumps({"name": "Ada", "age": 36}))
    r = verify(
        VerifySample("extraction", request_payload=_json_req(), response_payload=resp)
    )
    assert r.reward == 1.0


def test_structured_demanded_prose_only_scores_0() -> None:
    resp = _anthropic_text("The name is Ada and she is 36.")  # prose, but JSON demanded
    r = verify(
        VerifySample("extraction", request_payload=_json_req(), response_payload=resp)
    )
    assert r.reward == 0.0


def test_extraction_no_demand_no_gold_defers_not_zero() -> None:
    # NOT a constant 0: unverifiable without a structure-demand or gold → Council
    r = verify(
        VerifySample("extraction", response_payload=_anthropic_text("The name is Ada."))
    )
    assert r.reward is None and r.reward_source == ""
    # and a bare JSON value without demand/gold is ALSO not a free 1 anymore
    r2 = verify(VerifySample("extraction", response_payload=_anthropic_text("48291")))
    assert r2.reward is None


def test_extraction_schema_missing_required_key_scores_0() -> None:
    resp = _anthropic_text(json.dumps({"name": "Ada"}))  # missing age, schema demanded
    r = verify(
        VerifySample("extraction", request_payload=_json_req(), response_payload=resp)
    )
    assert r.reward == 0.0 and any("missing_keys" in e for e in r.evidence)


# ── AIN-547 sibling audit: other verifiers are correctness-over-format ────────


def test_audit_code_scores_correct_code_amid_prose() -> None:
    # code verifier extracts the fenced block — surrounding prose doesn't change it
    resp = _anthropic_text(
        "Sure! Here's the function:\n```python\ndef f(): return 1\n```\nHope that helps."
    )
    assert verify(VerifySample("code", response_payload=resp)).reward == 1.0
    # and it DEFERS (not 0) when there's no code block — never penalises a non-miss
    assert (
        verify(
            VerifySample("code", response_payload=_anthropic_text("Could you clarify?"))
        ).reward
        is None
    )


def test_audit_answer_numeric_is_format_robust() -> None:
    for got in ("42", "42.0", "$42", "The answer is 42.", "= 42"):
        r = verify(
            VerifySample(
                "reasoning", response_payload=_anthropic_text(got), expected=42
            )
        )
        assert r.reward == 1.0, got


# ── tool_use (intrinsic, partial) ────────────────────────────────────────────


def test_tool_wellformed_call_scores_1() -> None:
    req = {
        "tools": [
            {
                "function": {
                    "name": "get_weather",
                    "parameters": {"type": "object", "required": ["city"]},
                }
            }
        ]
    }
    resp = _openai(
        tool_calls=[
            {
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "Jakarta"}),
                }
            }
        ]
    )
    r = verify(VerifySample("tool_use", request_payload=req, response_payload=resp))
    assert r.reward == 1.0


def test_tool_unknown_name_scores_0() -> None:
    req = {
        "tools": [
            {"function": {"name": "get_weather", "parameters": {"type": "object"}}}
        ]
    }
    resp = _openai(
        tool_calls=[{"function": {"name": "launch_rocket", "arguments": "{}"}}]
    )
    r = verify(VerifySample("tool_use", request_payload=req, response_payload=resp))
    assert r.reward == 0.0


def test_tool_no_call_defers() -> None:
    r = verify(VerifySample("tool_use", response_payload=_openai("Sure, it's sunny.")))
    assert r.reward is None


# ── reasoning / answer (reference) ───────────────────────────────────────────


def test_answer_numeric_match_scores_1() -> None:
    resp = _anthropic_text("Working through it... Final answer: 42")
    r = verify(VerifySample("reasoning", response_payload=resp, expected="42"))
    assert r.reward == 1.0 and r.mode == "reference"


def test_answer_mismatch_scores_0() -> None:
    resp = _anthropic_text("Final answer: 41")
    r = verify(VerifySample("reasoning", response_payload=resp, expected=42))
    assert r.reward == 0.0


def test_reasoning_without_gold_defers() -> None:
    resp = _anthropic_text("Final answer: 42")
    r = verify(VerifySample("reasoning", response_payload=resp))  # no expected
    assert r.reward is None and r.verifiable is False


# ── dispatcher: subjective → Council ─────────────────────────────────────────


def test_subjective_tasks_defer_to_council() -> None:
    for subj in ("chat", "embed", "general", None, "bogus"):
        r = verify(VerifySample(subj, response_payload=_anthropic_text("anything")))
        assert r.reward is None and r.verifier == "none" and r.mode == "none"


# ── THE substantive-not-liveness invariant (v2 §1) ───────────────────────────


def test_empty_200_body_never_scores_1() -> None:
    """A '200 OK' with an empty/garbage body must never earn reward 1.0 — reward
    comes from a content check, never liveness."""
    empty_shapes = [
        _openai(""),
        _anthropic_text(""),
        {"choices": [{"message": {"content": ""}}]},
        {},  # not even a recognizable body
    ]
    for tt in CANONICAL_TASK_TYPES:
        for resp in empty_shapes:
            r = verify(VerifySample(tt, response_payload=resp, expected="x"))
            assert r.reward in (None, 0.0), f"{tt} empty body scored {r.reward}"


# ── sizing ───────────────────────────────────────────────────────────────────


def test_histogram_buckets_and_pct() -> None:
    hist = verifiability_histogram(["code", "code", "chat", "reasoning"])
    assert hist["verifiable"]["count"] == 2
    assert hist["partial"]["count"] == 1
    assert hist["subjective"]["count"] == 1
    assert round(sum(b["pct"] for b in hist.values())) == 100


def test_sizing_sql_covers_all_seven_types_and_uses_step0_stamp() -> None:
    for tt in CANONICAL_TASK_TYPES:
        assert f"'{tt}'" in VERIFIABILITY_SIZING_SQL
    assert "NOT exclude_from_training" in VERIFIABILITY_SIZING_SQL


def test_verify_rows_batch() -> None:
    rows = [
        {"task_type": "code", "response_payload": _openai("```python\nx=1\n```")},
        {"task_type": "chat", "response_payload": _anthropic_text("hi")},
    ]
    results = verify_rows(rows)
    assert results[0].reward == 1.0
    assert results[1].reward is None
