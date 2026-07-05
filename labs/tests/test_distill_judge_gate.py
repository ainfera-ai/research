"""AIN-561 · distilled-judge ship gate — agreement-vs-Council contract (fixture-verified)."""

from __future__ import annotations

from labs.distill_judge_gate import evaluate_distilled_judge


def test_perfect_agreement_ships() -> None:
    pairs = [("pass", "pass"), ("fail", "fail"), ("pass", "pass"), ("fail", "fail")]
    r = evaluate_distilled_judge(pairs)
    assert r.raw_agreement == 1.0 and r.cohen_kappa == 1.0 and r.ships is True


def test_empty_does_not_ship() -> None:
    r = evaluate_distilled_judge([])
    assert r.cohen_kappa is None and r.ships is False  # never assume agreement


def test_kappa_gate_boundary() -> None:
    # 6/8 agree on balanced labels ⇒ Cohen's κ = 0.5: below the 0.60 default → no ship.
    pairs = [("pass", "pass")] * 3 + [("fail", "fail")] * 3 + [("pass", "fail"), ("fail", "pass")]
    assert abs(evaluate_distilled_judge(pairs).cohen_kappa - 0.5) < 1e-9
    assert evaluate_distilled_judge(pairs).ships is False
    assert evaluate_distilled_judge(pairs, min_kappa=0.4).ships is True


def test_anticorrelated_judge_does_not_ship() -> None:
    # a judge that mostly disagrees has κ ≤ 0 — must never ship as the bulk labeler
    pairs = [("pass", "fail"), ("fail", "pass"), ("pass", "fail"), ("fail", "pass")]
    r = evaluate_distilled_judge(pairs, min_kappa=0.0)
    assert r.ships is False
