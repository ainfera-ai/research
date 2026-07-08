"""savings_holdout.py — AIN-542 DR-OPE wired to live savings holdout.

The savings holdout measures whether the ainfera router delivers *done-and-cheaper*
outcomes vs the agreed baseline, using doubly-robust off-policy evaluation on
LOGGED routing_outcomes traffic.

Primary metric:   success on outcome_status (1.0 if 'succeeded', 0.0 otherwise)
Secondary metrics: cost_actual_usd, observed_latency_ms

The DR-OPE math lives in dr_ope.py (pure, tested). This module is the LIVE
wiring: it reads routing_outcomes rows from the production DB, constructs
LoggedSample objects with the correct propensity, and calls evaluate_policy.

Key fix — propensity logging bug (0/3789 populated):
  The routing_brain computes chosen_propensity under
  _propensity_logging_enabled() (default ON since 2026-06-19, F-R7 go-live).
  Rows written BEFORE the flag was enabled have NULL chosen_propensity.
  Rows from deterministic decisions (single survivor, ε=0) also have
  propensity 1.0 but may be NULL in the column.
  
  AIN-549: a `deterministic` decision_type row with NULL chosen_propensity
  is treated as π_b=1.0 (well-defined), NOT as "no propensity". Any other
  decision_type with NULL propensity is excluded (not weightable).
  
  This module enforces that rule when building samples — so the 0/3789
  gap is closed for all deterministic decisions (the vast majority of traffic
  under ε=0 or single-survivor), and only genuinely stochastic rows without
  logged propensity are excluded.

Bootstrap CI + ESS reporting come from dr_ope.evaluate_policy (paired
bootstrap on the lift, Kish ESS on the importance weights).

References:
  dr_ope.py            — pure DR-OPE math
  labs_ope_eval.py     — the existing candidate-eval runner (promotion gate)
  routing_propensity.py — ε-floor propensity computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from labs.dr_ope import LoggedSample, OPEResult, evaluate_policy

log = logging.getLogger(__name__)

# ── SQL: fetch logged routing_outcomes for DR-OPE ────────────────────────────
#
# AIN-549: consume ONLY correctly-weightable rows:
#   - deterministic decisions: π_b(chosen) = 1.0 (propensity may be NULL)
#   - stochastic/explore decisions: MUST carry chosen_propensity
# Rows that are stochastic without a logged vector are excluded (never
# silently treated as no-correction).
#
# traffic_class IN ('customer', 'fleet') — DR-OPE on real traffic only
# (AIN-521: exclude internal_eval + internal_probe).
# judge_status = 'labeled' AND reward IS NOT NULL — we need observed reward.
# outcome_status IS NOT NULL — completion fields must be written.

SAVINGS_HOLDOUT_SQL = """
SELECT
    ro.id,
    ro.task_type,
    ro.chosen_model_slug,
    ro.outcome_status,
    ro.cost_actual_usd,
    ro.observed_latency_ms,
    ro.reward,
    ro.chosen_propensity,
    ro.decision_type,
    ro.baseline_model_slug,
    ro.cost_baseline_usd,
    ro.holdout_arm,
    ro.ab_arm,
    ro.traffic_class,
    ro.created_at
FROM public.routing_outcomes ro
WHERE ro.judge_status = 'labeled'
  AND ro.reward IS NOT NULL
  AND ro.outcome_status IS NOT NULL
  AND ro.traffic_class IN ('customer', 'fleet')
  AND (
      ro.decision_type = 'deterministic'
      OR ro.chosen_propensity IS NOT NULL
  )
  AND ro.created_at >= now() - ($1::int * interval '1 day')
ORDER BY ro.created_at ASC
"""

# ── reward → [0,1] mapping ───────────────────────────────────────────────────
#
# Primary metric: success on outcome_status.
#   outcome_status = 'succeeded' → reward = 1.0
#   outcome_status = 'failed' / 'rejected' / 'error' → reward = 0.0
# If the row already has a judge-based reward in [0,1], use that instead
# (it's a finer-grained signal — the judge score maps to [0,1] via (score-1)/4).
# The reward column is the authoritative signal: it's either verify()-sourced
# (Tier-A, 0.0/1.0) or judge-sourced (Tier-B, continuous in [0,1]).


def _reward_from_row(row: dict[str, Any]) -> float:
    """Extract the reward in [0,1] from a routing_outcomes row.

    Prefers the `reward` column (already in [0,1], sourced from verify() or
    judge). Falls back to outcome_status → binary success.
    """
    raw_reward = row.get("reward")
    if raw_reward is not None:
        r = float(raw_reward)
        if 0.0 <= r <= 1.0:
            return r
    # Fallback: binary from outcome_status
    status = (row.get("outcome_status") or "").lower()
    return 1.0 if status == "succeeded" else 0.0


def _propensity_from_row(row: dict[str, Any]) -> float:
    """Extract the behaviour-policy propensity π_b(chosen) from a row.

    AIN-549: a `deterministic` decision has π_b(chosen) = 1.0 even if the
    column is NULL (the pick was deterministic → propensity is trivially 1).
    A stochastic/explore row MUST have a non-null chosen_propensity (the SQL
    already enforces this). Any row with propensity ≤ 0 gets weight 0 in
    the DR estimator (IPS term dropped, direct-method kept).
    """
    p = row.get("chosen_propensity")
    if p is not None:
        return float(p)
    if (row.get("decision_type") or "").lower() == "deterministic":
        return 1.0
    # Should not reach here (SQL filters these out), but fail safe
    return 0.0


@dataclass(frozen=True)
class SavingsMetric:
    """One metric from the savings holdout."""

    name: str
    ope: OPEResult
    description: str


@dataclass(frozen=True)
class SavingsHoldoutResult:
    """Full result of the DR-OPE savings holdout run."""

    primary: SavingsMetric  # success on outcome_status
    cost: SavingsMetric | None  # cost_actual_usd (if populated)
    latency: SavingsMetric | None  # observed_latency_ms (if populated)
    n_samples: int
    n_propensity_logged: int
    n_deterministic: int
    n_stochastic: int
    notes: str = ""


def build_samples(rows: list[dict[str, Any]]) -> list[LoggedSample]:
    """Build LoggedSample objects from routing_outcomes rows.

    Uses the reward column (sourced from verify() or judge) as the primary
    reward signal, and the chosen_propensity (with AIN-549 deterministic
    fallback) as the behaviour-policy propensity.
    """
    samples: list[LoggedSample] = []
    for row in rows:
        task_type = row.get("task_type") or "unknown"
        action = row.get("chosen_model_slug") or "unknown"
        reward = _reward_from_row(row)
        propensity = _propensity_from_row(row)
        samples.append(
            LoggedSample(
                task_type=task_type,
                action=action,
                reward=reward,
                propensity=propensity,
            )
        )
    return samples


def build_cost_samples(rows: list[dict[str, Any]]) -> list[LoggedSample]:
    """Build LoggedSample objects using cost_actual_usd as the reward signal.

    For cost savings, lower is better. We invert: reward = 1 - normalized_cost.
    The normalization uses the max observed cost in the sample set (so the
    cheapest sample gets reward ≈ 1.0 and the most expensive gets ≈ 0.0).
    """
    costs = [float(r.get("cost_actual_usd") or 0.0) for r in rows]
    if not costs or max(costs) <= 0:
        return []
    max_cost = max(costs)
    samples: list[LoggedSample] = []
    for row, cost in zip(rows, costs, strict=True):
        if cost <= 0:
            continue  # skip rows without cost data
        task_type = row.get("task_type") or "unknown"
        action = row.get("chosen_model_slug") or "unknown"
        # Invert + normalize: cheaper → higher reward
        reward = 1.0 - (cost / max_cost)
        propensity = _propensity_from_row(row)
        samples.append(
            LoggedSample(
                task_type=task_type,
                action=action,
                reward=max(0.0, min(1.0, reward)),
                propensity=propensity,
            )
        )
    return samples


def build_latency_samples(rows: list[dict[str, Any]]) -> list[LoggedSample]:
    """Build LoggedSample objects using observed_latency_ms as the reward signal.

    For latency, lower is better. We invert: reward = 1 - normalized_latency.
    """
    latencies = [
        float(r.get("observed_latency_ms") or 0)
        for r in rows
        if r.get("observed_latency_ms") is not None
    ]
    if not latencies or max(latencies) <= 0:
        return []
    max_lat = max(latencies)
    samples: list[LoggedSample] = []
    for row in rows:
        lat = row.get("observed_latency_ms")
        if lat is None or lat <= 0:
            continue
        task_type = row.get("task_type") or "unknown"
        action = row.get("chosen_model_slug") or "unknown"
        reward = 1.0 - (float(lat) / max_lat)
        propensity = _propensity_from_row(row)
        samples.append(
            LoggedSample(
                task_type=task_type,
                action=action,
                reward=max(0.0, min(1.0, reward)),
                propensity=propensity,
            )
        )
    return samples


def _target_policy_uniform(samples: list[LoggedSample]) -> dict[str, dict[str, float]]:
    """Build a uniform target policy over all actions seen per task_type.

    This is the simplest target: π(a|x) = 1/|A(x)| for each action a in
    task_type x. The DR estimator then tells you the expected value of
    picking uniformly at random vs the logging policy's observed value.
    """
    actions_by_type: dict[str, set[str]] = {}
    for s in samples:
        actions_by_type.setdefault(s.task_type, set()).add(s.action)
    return {
        tt: {a: 1.0 / len(actions) for a in actions}
        for tt, actions in actions_by_type.items()
    }


def _q_hat_group_mean(samples: list[LoggedSample]) -> dict[tuple[str, str], float]:
    """Build q̂(x,a) = group mean reward per (task_type, action)."""
    sums: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    for s in samples:
        key = (s.task_type, s.action)
        sums[key] = sums.get(key, 0.0) + s.reward
        counts[key] = counts.get(key, 0) + 1
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def evaluate_savings_holdout(
    rows: list[dict[str, Any]],
    *,
    target_probs: dict[str, dict[str, float]] | None = None,
    q_hat: dict[tuple[str, str], float] | None = None,
    weight_clip: float | None = 10.0,
    bootstrap: int = 2000,
    seed: int = 0,
) -> SavingsHoldoutResult:
    """Run the DR-OPE savings holdout on logged routing_outcomes rows.

    Primary metric:   success on outcome_status (reward column)
    Secondary metrics: cost_actual_usd (inverted+normalized), observed_latency_ms (inverted+normalized)

    Returns the DR-OPE result for each metric, with bootstrap CI + ESS.

    If target_probs or q_hat are not provided, defaults are used:
    - target_probs: uniform over all actions per task_type
    - q_hat: group mean reward per (task_type, action)
    """
    samples = build_samples(rows)
    if not samples:
        primary = SavingsMetric(
            name="success",
            ope=OPEResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0),
            description="No samples — empty holdout",
        )
        return SavingsHoldoutResult(
            primary=primary,
            cost=None,
            latency=None,
            n_samples=0,
            n_propensity_logged=0,
            n_deterministic=0,
            n_stochastic=0,
            notes="No weightable rows in the lookback window.",
        )

    # Build target policy + reward model if not provided
    tp = target_probs or _target_policy_uniform(samples)
    qh = q_hat or _q_hat_group_mean(samples)

    # Primary: success rate
    ope_primary = evaluate_policy(
        samples, tp, qh, weight_clip=weight_clip, bootstrap=bootstrap, seed=seed
    )
    primary = SavingsMetric(
        name="success",
        ope=ope_primary,
        description="DR-OPE on outcome_status success (reward column)",
    )

    # Secondary: cost
    cost_samples = build_cost_samples(rows)
    cost_metric = None
    if cost_samples:
        tp_cost = target_probs or _target_policy_uniform(cost_samples)
        qh_cost = q_hat or _q_hat_group_mean(cost_samples)
        ope_cost = evaluate_policy(
            cost_samples, tp_cost, qh_cost,
            weight_clip=weight_clip, bootstrap=bootstrap, seed=seed + 1,
        )
        cost_metric = SavingsMetric(
            name="cost",
            ope=ope_cost,
            description="DR-OPE on cost_actual_usd (inverted+normalized: cheaper=higher reward)",
        )

    # Secondary: latency
    latency_samples = build_latency_samples(rows)
    latency_metric = None
    if latency_samples:
        tp_lat = target_probs or _target_policy_uniform(latency_samples)
        qh_lat = q_hat or _q_hat_group_mean(latency_samples)
        ope_lat = evaluate_policy(
            latency_samples, tp_lat, qh_lat,
            weight_clip=weight_clip, bootstrap=bootstrap, seed=seed + 2,
        )
        latency_metric = SavingsMetric(
            name="latency",
            ope=ope_lat,
            description="DR-OPE on observed_latency_ms (inverted+normalized: faster=higher reward)",
        )

    # Diagnostics
    n_prop = sum(1 for r in rows if r.get("chosen_propensity") is not None)
    n_det = sum(1 for r in rows if (r.get("decision_type") or "").lower() == "deterministic")
    n_stoch = len(rows) - n_det

    return SavingsHoldoutResult(
        primary=primary,
        cost=cost_metric,
        latency=latency_metric,
        n_samples=len(samples),
        n_propensity_logged=n_prop,
        n_deterministic=n_det,
        n_stochastic=n_stoch,
    )


__all__ = [
    "SAVINGS_HOLDOUT_SQL",
    "SavingsMetric",
    "SavingsHoldoutResult",
    "build_samples",
    "build_cost_samples",
    "build_latency_samples",
    "evaluate_savings_holdout",
]
