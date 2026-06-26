"""AIN-561 · distilled-judge cost report — API-calls/cost-avoided per nightly run."""

from __future__ import annotations

import pytest

from labs.distill_cost_report import build_cost_report


def test_basic_avoided_math() -> None:
    r = build_cost_report(
        run_date="2026-06-25",
        items_labeled=1000,
        teacher_unit_usd=0.003,
        student_unit_usd=0.0001,
    )
    assert r.teacher_calls_avoided == 1000  # 1:1 replacement on the bulk path
    assert r.teacher_cost_avoided_usd == 3.0
    assert r.student_cost_incurred_usd == 0.1
    assert r.net_cost_avoided_usd == 2.9
    assert r.net_cost_avoided_pct == round(100.0 * 2.9 / 3.0, 2)  # ~96.67


def test_zero_items_is_inert_no_div_by_zero() -> None:
    r = build_cost_report(
        run_date="d", items_labeled=0, teacher_unit_usd=0.003, student_unit_usd=0.0
    )
    assert r.teacher_calls_avoided == 0
    assert r.net_cost_avoided_usd == 0.0
    assert r.net_cost_avoided_pct == 0.0  # no baseline → floored to 0, not a ZeroDivisionError


def test_deterministic() -> None:
    kw = dict(run_date="d", items_labeled=42, teacher_unit_usd=0.003, student_unit_usd=0.0001)
    assert build_cost_report(**kw) == build_cost_report(**kw)


@pytest.mark.parametrize(
    "bad",
    [
        dict(run_date="d", items_labeled=-1, teacher_unit_usd=0.003, student_unit_usd=0.0),
        dict(run_date="d", items_labeled=1, teacher_unit_usd=-0.003, student_unit_usd=0.0),
        dict(run_date="d", items_labeled=1, teacher_unit_usd=0.003, student_unit_usd=-0.1),
    ],
)
def test_rejects_negative_inputs(bad: dict) -> None:
    with pytest.raises(ValueError):
        build_cost_report(**bad)
