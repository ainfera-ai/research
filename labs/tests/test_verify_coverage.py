"""AIN-624 · κ-coverage screening harness unit tests.

Fixtures only (no DB, no network): the harness logic is validated offline; the
*informative* coverage number needs live traffic (the tap). Pins the
SCREENING-ONLY invariant — the report carries no promotion verdict.
"""

from __future__ import annotations

import json

from labs.verify_coverage import (
    SCREENING_ONLY,
    CoverageReport,
    FamilyCoverage,
    compute_coverage,
    format_report,
)


# ── payload builders (mirror test_verify_harness) ────────────────────────────


def _openai(text: str = "", tool_calls=None) -> dict:
    msg: dict = {"content": text}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"message": msg}]}


def _code(src: str) -> dict:
    return {"task_type": "code", "response_payload": _openai(f"```python\n{src}\n```")}


def _json_req() -> dict:
    return {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"schema": {"type": "object", "required": ["name", "age"]}},
        }
    }


def _extraction(body: str) -> dict:
    return {
        "task_type": "extraction",
        "request_payload": _json_req(),
        "response_payload": _openai(body),
    }


def _tool_req() -> dict:
    return {
        "tools": [
            {"function": {"name": "get_weather", "parameters": {"type": "object", "required": ["city"]}}}
        ]
    }


def _tool_call(args: dict) -> dict:
    return {
        "task_type": "tool_use",
        "request_payload": _tool_req(),
        "response_payload": _openai(
            "", tool_calls=[{"function": {"name": "get_weather", "arguments": json.dumps(args)}}]
        ),
    }


# ── coverage = reach over the intrinsic-eligible population ───────────────────


def test_all_code_failures_are_caught_full_coverage() -> None:
    rows = [_code("def f(:\n    return") for _ in range(5)]  # all syntax errors
    rep = compute_coverage(rows)
    code = rep.families["code_exec"]
    assert code.intrinsic is True
    assert code.n_eligible == 5 and code.n_graded == 5 and code.n_fail == 5
    assert code.coverage == 1.0 and code.fail_rate == 1.0
    assert rep.intrinsic_coverage == 1.0
    assert rep.n_fail_total == 5


def test_deferral_lowers_coverage() -> None:
    # 2 gradable (syntax error) + 3 no-code-block (defer) → 2/5 reach.
    rows = [_code("def f(:\n  x")] * 2 + [
        {"task_type": "code", "response_payload": _openai("I'd refactor the loop.")}
        for _ in range(3)
    ]
    rep = compute_coverage(rows)
    code = rep.families["code_exec"]
    assert code.n_eligible == 5 and code.n_graded == 2 and code.n_deferred == 3
    assert code.coverage == 0.4
    assert rep.intrinsic_coverage == 0.4


def test_mixed_pass_fail_failrate() -> None:
    rows = [
        _code("def ok():\n    return 1"),  # pass
        _code("def ok2():\n    return 2"),  # pass
        _code("def bad(:\n    return"),  # fail
    ]
    rep = compute_coverage(rows)
    code = rep.families["code_exec"]
    assert code.n_pass == 2 and code.n_fail == 1
    assert code.fail_rate == 1 / 3
    assert rep.fail_rate_total == 1 / 3


def test_extraction_json_valid_and_invalid() -> None:
    rows = [
        _extraction(json.dumps({"name": "a", "age": 5})),  # pass
        _extraction("definitely not json"),  # fail (json demanded)
    ]
    rep = compute_coverage(rows)
    schema = rep.families["schema_match"]
    assert schema.intrinsic is True
    assert schema.n_graded == 2 and schema.n_pass == 1 and schema.n_fail == 1


def test_tool_wellformed_vs_malformed() -> None:
    rows = [
        _tool_call({"city": "Jakarta"}),  # conforms → pass
        _tool_call({}),  # missing required 'city' → fail
    ]
    rep = compute_coverage(rows)
    tool = rep.families["tool_result"]
    assert tool.intrinsic is True
    assert tool.n_pass == 1 and tool.n_fail == 1 and tool.coverage == 1.0


# ── reference vs intrinsic: only intrinsic rows count toward the headline ─────


def test_reference_family_with_gold_grades_but_excluded_from_intrinsic_denom() -> None:
    rows = [
        {"task_type": "reasoning", "expected": "42", "response_payload": _openai("The answer is 42.")},
    ]
    rep = compute_coverage(rows)
    ans = rep.families["answer_check"]
    assert ans.intrinsic is False
    assert ans.n_graded == 1 and ans.n_pass == 1  # reference check fired (gold present)
    # graded overall, but NOT part of the no-gold intrinsic coverage denominator.
    assert rep.n_graded_total == 1
    assert rep.n_intrinsic_eligible == 0 and rep.intrinsic_coverage is None


def test_reference_family_without_gold_defers() -> None:
    rows = [{"task_type": "reasoning", "response_payload": _openai("Some reasoning, no gold.")}]
    rep = compute_coverage(rows)
    assert rep.families["answer_check"].n_deferred == 1
    assert rep.n_graded_total == 0


def test_subjective_rows_excluded_from_intrinsic_denom() -> None:
    rows = [
        {"task_type": "chat", "response_payload": _openai("hi there!")},
        _code("def ok():\n    return 1"),  # 1 intrinsic-eligible
    ]
    rep = compute_coverage(rows)
    assert rep.families["none"].intrinsic is False and rep.families["none"].n_deferred == 1
    assert rep.n_intrinsic_eligible == 1 and rep.intrinsic_coverage == 1.0


# ── honesty: vacuous denominators report None, never a fabricated 0 ───────────


def test_empty_corpus_is_vacuous_not_zero() -> None:
    rep = compute_coverage([])
    assert rep.n_rows == 0
    assert rep.intrinsic_coverage is None
    assert rep.fail_rate_total is None


def test_all_pass_failrate_is_zero_not_none() -> None:
    rep = compute_coverage([_code("x = 1")])
    assert rep.n_graded_total == 1 and rep.n_fail_total == 0
    assert rep.fail_rate_total == 0.0  # graded with no failures → 0.0, distinct from None


# ── SCREENING-ONLY invariant (the load-bearing rule) ──────────────────────────


def test_screening_only_no_promotion_surface() -> None:
    assert SCREENING_ONLY is True
    rep = compute_coverage([_code("x = 1")])
    banned = ("hold", "gate", "promot", "kappa_valid")
    for field_name in CoverageReport.__dataclass_fields__:
        assert not any(b in field_name.lower() for b in banned), field_name
    for field_name in FamilyCoverage.__dataclass_fields__:
        assert not any(b in field_name.lower() for b in banned), field_name
    assert not hasattr(rep, "promotion_hold")


def test_report_renders_without_verdict_line() -> None:
    rep = compute_coverage([_code("def bad(:\n  x"), _code("ok = 1")])
    text = format_report(rep)
    assert "SCREENING-ONLY" in text and "no promotion gate" in text
    low = text.lower()
    assert "promote" not in low and "promotion_hold" not in low
