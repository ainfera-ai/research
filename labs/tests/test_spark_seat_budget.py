"""AIN-548 · Spark Labs seat budgeting contract (128 GB envelope, fixture-verified)."""

from __future__ import annotations

from labs.spark_seat_budget import SeatRequest, plan_seats


def test_admits_until_envelope_full() -> None:
    reqs = [
        SeatRequest("competitor_loop", 40.0, 2.0),
        SeatRequest("distill", 60.0, 5.0),
        SeatRequest("refit", 20.0, 0.5),
        SeatRequest("oversized", 50.0, 1.0),  # 120 used; 50 > 8 free → rejected
    ]
    plan = plan_seats(reqs, gpu_cost_per_hour=2.0)
    assert plan.admitted == ["competitor_loop", "distill", "refit"]
    assert plan.rejected == ["oversized"]
    assert plan.used_gb == 120.0 and plan.free_gb == 8.0
    assert plan.utilisation == round(120.0 / 128.0, 6)
    assert plan.total_gpu_hours == 7.5
    assert plan.total_cost_usd == 15.0  # 7.5h * $2


def test_seat_larger_than_envelope_is_rejected() -> None:
    plan = plan_seats([SeatRequest("huge", 256.0, 1.0)])
    assert plan.admitted == [] and plan.rejected == ["huge"] and plan.used_gb == 0.0


def test_empty_plan() -> None:
    plan = plan_seats([])
    assert plan.admitted == [] and plan.utilisation == 0.0 and plan.total_cost_usd == 0.0
