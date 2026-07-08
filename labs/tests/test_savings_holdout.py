"""savings_holdout.py — DR-OPE savings holdout tests (AIN-542).

Tests the live wiring: row → LoggedSample construction, propensity fallback,
multi-metric evaluation (success + cost + latency), and the AIN-549 deterministic
propensity rule that closes the 0/3789 gap.
"""

from __future__ import annotations

from labs.dr_ope import OPEResult
from labs.savings_holdout import (
    SAVINGS_HOLDOUT_SQL,
    SavingsHoldoutResult,
    build_cost_samples,
    build_latency_samples,
    build_samples,
    evaluate_savings_holdout,
)


# ── row → LoggedSample construction ──────────────────────────────────────────


def _row(
    *,
    task_type="code",
    slug="gpt-5-5",
    reward: float | None = 0.8,
    propensity=None,
    decision_type="deterministic",
    outcome_status="succeeded",
    cost=0.02,
    latency=500,
    _id="row-1",
):
    return {
        "id": "row-1",
        "task_type": task_type,
        "chosen_model_slug": slug,
        "outcome_status": outcome_status,
        "cost_actual_usd": cost,
        "observed_latency_ms": latency,
        "reward": reward,
        "chosen_propensity": propensity,
        "decision_type": decision_type,
        "baseline_model_slug": None,
        "cost_baseline_usd": None,
        "holdout_arm": None,
        "ab_arm": None,
        "traffic_class": "customer",
        "created_at": "2026-07-03T00:00:00Z",
    }


def test_build_samples_extracts_reward_and_propensity():
    rows = [_row(reward=0.8, propensity=0.5, decision_type="explore")]
    samples = build_samples(rows)
    assert len(samples) == 1
    s = samples[0]
    assert s.task_type == "code"
    assert s.action == "gpt-5-5"
    assert s.reward == 0.8
    assert s.propensity == 0.5


def test_deterministic_null_propensity_treated_as_one():
    """AIN-549: a deterministic decision with NULL propensity → π_b = 1.0,
    NOT 0.0. This is the fix for the 0/3789 populated bug."""
    rows = [_row(propensity=None, decision_type="deterministic")]
    samples = build_samples(rows)
    assert samples[0].propensity == 1.0  # deterministic → π_b = 1


def test_stochastic_null_propensity_treated_as_zero():
    """A stochastic row without logged propensity → π_b = 0 → weight 0 (IPS
    dropped, direct-method only). The SQL already filters these out, but
    build_samples is defensive."""
    rows = [_row(propensity=None, decision_type="explore")]
    samples = build_samples(rows)
    assert samples[0].propensity == 0.0


def test_explicit_propensity_preserved():
    rows = [_row(propensity=0.3, decision_type="explore")]
    samples = build_samples(rows)
    assert samples[0].propensity == 0.3


def test_reward_fallback_to_outcome_status():
    """If reward column is NULL, fall back to outcome_status → binary."""
    rows = [_row(reward=None, outcome_status="succeeded")]
    samples = build_samples(rows)
    assert samples[0].reward == 1.0

    rows2 = [_row(reward=None, outcome_status="failed")]
    samples2 = build_samples(rows2)
    assert samples2[0].reward == 0.0


# ── cost + latency secondary metrics ─────────────────────────────────────────


def test_cost_samples_inverted_normalized():
    """Cost: cheaper → higher reward. Most expensive → reward ≈ 0."""
    rows = [
        _row(_id="r1", cost=0.01, slug="cheap-model"),
        _row(_id="r2", cost=0.10, slug="expensive-model"),
    ]
    samples = build_cost_samples(rows)
    assert len(samples) == 2
    # cheap model (0.01) → reward = 1 - 0.01/0.10 = 0.9
    # expensive model (0.10) → reward = 1 - 0.10/0.10 = 0.0
    assert samples[0].reward > samples[1].reward
    assert abs(samples[0].reward - 0.9) < 0.01
    assert abs(samples[1].reward - 0.0) < 0.01


def test_latency_samples_inverted_normalized():
    """Latency: faster → higher reward."""
    rows = [
        _row(_id="r1", latency=100, slug="fast-model"),
        _row(_id="r2", latency=2000, slug="slow-model"),
    ]
    samples = build_latency_samples(rows)
    assert len(samples) == 2
    assert samples[0].reward > samples[1].reward


def test_cost_samples_skip_zero_cost():
    rows = [
        _row(_id="r1", cost=0.0),
        _row(_id="r2", cost=0.05),
    ]
    samples = build_cost_samples(rows)
    assert len(samples) == 1  # zero-cost row skipped


# ── evaluate_savings_holdout end-to-end ──────────────────────────────────────


def test_evaluate_savings_holdout_returns_all_metrics():
    rows = [
        _row(_id="r1", slug="gpt-5-5", reward=0.8, propensity=0.5, decision_type="explore", cost=0.02, latency=500),
        _row(_id="r2", slug="gpt-5-5", reward=0.6, propensity=0.5, decision_type="explore", cost=0.03, latency=800),
        _row(_id="r3", slug="claude-opus-4-7", reward=0.9, propensity=0.5, decision_type="explore", cost=0.05, latency=300),
    ]
    result = evaluate_savings_holdout(rows, seed=42)
    assert isinstance(result, SavingsHoldoutResult)
    assert result.n_samples == 3
    assert result.primary.name == "success"
    assert result.primary.ope.n == 3
    assert result.cost is not None
    assert result.latency is not None
    assert result.cost.ope.n == 3
    assert result.latency.ope.n == 3


def test_evaluate_savings_holdout_empty_rows():
    result = evaluate_savings_holdout([])
    assert result.n_samples == 0
    assert result.primary.ope.v_dr == 0.0
    assert result.cost is None
    assert result.latency is None


def test_evaluate_savings_holdout_deterministic_only():
    """All deterministic rows with NULL propensity — should still work (π_b=1)."""
    rows = [
        _row(_id="r1", propensity=None, decision_type="deterministic", reward=0.8),
        _row(_id="r2", propensity=None, decision_type="deterministic", reward=0.6),
    ]
    result = evaluate_savings_holdout(rows, seed=1)
    assert result.n_samples == 2
    assert result.n_deterministic == 2
    assert result.n_stochastic == 0
    # With π_b=1 and uniform target, ESS should be high (weights ≈ 1)
    assert result.primary.ope.ess > 0


def test_sql_excludes_internal_eval_and_internal_probe():
    """The SQL must exclude internal_eval + internal_probe traffic_class."""
    assert "internal_eval" not in SAVINGS_HOLDOUT_SQL
    assert "internal_probe" not in SAVINGS_HOLDOUT_SQL
    assert "'customer', 'fleet'" in SAVINGS_HOLDOUT_SQL


def test_sql_requires_labeled_reward():
    """The SQL must require judge_status='labeled' AND reward IS NOT NULL."""
    assert "judge_status = 'labeled'" in SAVINGS_HOLDOUT_SQL
    assert "reward IS NOT NULL" in SAVINGS_HOLDOUT_SQL


def test_sql_requires_outcome_status():
    """The SQL must require outcome_status IS NOT NULL (completion fields written)."""
    assert "outcome_status IS NOT NULL" in SAVINGS_HOLDOUT_SQL


def test_sql_enforces_ain549_weightability():
    """The SQL must enforce: deterministic OR chosen_propensity IS NOT NULL."""
    assert "decision_type = 'deterministic'" in SAVINGS_HOLDOUT_SQL
    assert "chosen_propensity IS NOT NULL" in SAVINGS_HOLDOUT_SQL


def test_deterministic_count_and_propensity_logged():
    rows = [
        _row(_id="r1", propensity=0.5, decision_type="explore"),
        _row(_id="r2", propensity=None, decision_type="deterministic"),
        _row(_id="r3", propensity=None, decision_type="deterministic"),
    ]
    result = evaluate_savings_holdout(rows, seed=1)
    assert result.n_deterministic == 2
    assert result.n_stochastic == 1
    assert result.n_propensity_logged == 1  # only r1 has explicit propensity
