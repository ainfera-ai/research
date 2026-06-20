"""DR-OPE — doubly-robust off-policy evaluation (AIN-542 D7/promotion gate)."""

from __future__ import annotations

import pytest

from labs.dr_ope import LoggedSample, evaluate_policy

# Logging policy on one context "t": 50/50 over action a (reward 0.8) and b (reward 0.4),
# so the on-policy value is 0.6.
PERFECT_Q = {("t", "a"): 0.8, ("t", "b"): 0.4}


def _samples(spec: list[tuple[str, float, float, int]]) -> list[LoggedSample]:
    out: list[LoggedSample] = []
    for action, reward, prop, count in spec:
        out += [LoggedSample("t", action, reward, prop) for _ in range(count)]
    return out


LOG = _samples([("a", 0.8, 0.5, 50), ("b", 0.4, 0.5, 50)])


def test_target_equals_logging_zero_lift() -> None:
    res = evaluate_policy(LOG, {"t": {"a": 0.5, "b": 0.5}}, PERFECT_Q, seed=1)
    assert res.v_dr == pytest.approx(0.6)
    assert res.v_logged == pytest.approx(0.6)
    assert res.lift == pytest.approx(0.0, abs=1e-9)
    assert res.ci_low <= 0.0 <= res.ci_high


def test_shifting_to_better_action_positive_lift() -> None:
    res = evaluate_policy(LOG, {"t": {"a": 1.0, "b": 0.0}}, PERFECT_Q, seed=1)
    assert res.v_dr == pytest.approx(0.8)
    assert res.lift == pytest.approx(0.2)
    assert res.ci_low > 0.0  # confidently better


def test_doubly_robust_rescues_wrong_q_hat() -> None:
    # q̂ is WRONG (0.5 everywhere) but the propensities μ are correct → DR still unbiased.
    wrong_q = {("t", "a"): 0.5, ("t", "b"): 0.5}
    res = evaluate_policy(LOG, {"t": {"a": 1.0, "b": 0.0}}, wrong_q, seed=1)
    assert res.v_dr == pytest.approx(0.8)  # IPS correction fixes the bad reward model


def test_ess_uniform_vs_degenerate() -> None:
    uniform = evaluate_policy(LOG, {"t": {"a": 0.5, "b": 0.5}}, PERFECT_Q, seed=1)
    assert uniform.ess == pytest.approx(100.0)  # all weights 1 → ESS = n
    degenerate = evaluate_policy(LOG, {"t": {"a": 1.0, "b": 0.0}}, PERFECT_Q, seed=1)
    assert degenerate.ess == pytest.approx(50.0)  # half the weights are 0
    assert degenerate.ess < degenerate.n


def test_weight_clip_bounds_ratio() -> None:
    rare = _samples([("a", 0.8, 0.01, 100)])  # μ=0.01, π=1.0 → ρ=100, clipped to 10
    res = evaluate_policy(rare, {"t": {"a": 1.0}}, PERFECT_Q, weight_clip=10.0, seed=1)
    assert res.mean_weight == pytest.approx(10.0)


def test_zero_propensity_no_division_error() -> None:
    s = _samples([("a", 0.8, 0.0, 10)])  # propensity 0 → weight 0, IPS term dropped
    res = evaluate_policy(s, {"t": {"a": 1.0}}, PERFECT_Q, seed=1)
    assert res.mean_weight == 0.0
    assert res.v_dr == pytest.approx(0.8)  # falls back to the direct estimate


def test_deterministic_and_ci_brackets_point() -> None:
    a = evaluate_policy(LOG, {"t": {"a": 0.7, "b": 0.3}}, PERFECT_Q, seed=9)
    b = evaluate_policy(LOG, {"t": {"a": 0.7, "b": 0.3}}, PERFECT_Q, seed=9)
    assert a == b
    assert a.ci_low <= a.lift <= a.ci_high


def test_empty_samples() -> None:
    res = evaluate_policy([], {"t": {"a": 1.0}}, PERFECT_Q)
    assert res.n == 0
    assert res.v_dr == 0.0 and res.lift == 0.0
