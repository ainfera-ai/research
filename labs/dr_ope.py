"""dr_ope.py — doubly-robust off-policy evaluation for candidate routing policies (DR-OPE, AIN-542).

The promotion gate's quantitative half: score a CANDIDATE policy on LOGGED traffic
*before* any canary, so you can say "policy v+1 is +3% reward (CI [+1%, +5%])" with a
confidence interval and no customer exposure.

Doubly-robust estimator (per logged sample i, context x = task_type, action a):

    V̂_DR = (1/n) Σ_i [ Σ_a π(a|x_i)·q̂(x_i,a)  +  ρ_i·(r_i − q̂(x_i,a_i)) ]
    ρ_i  = π(a_i|x_i) / μ(a_i|x_i)        (importance weight, optionally clipped)

  · π = TARGET (candidate) policy action distribution — e.g. the D7 alloc_weight.
  · μ = LOGGING (behaviour) policy propensity — logged per row (R7/AIN-462).
  · q̂ = a reward model (direct method) — e.g. the D1 q_empirical per cell.
  · r = observed reward in [0,1].

"Doubly robust": consistent if EITHER q̂ OR μ is correct. The direct term gives low
variance; the IPS correction removes q̂'s bias when μ is right (and vice-versa) — so a
wrong q̂ is rescued by the importance weights, and noisy weights are anchored by q̂.

Reported against the on-policy logged mean reward (the logging policy's value), with a
paired bootstrap CI on the LIFT and an effective-sample-size diagnostic (low ESS ⇒ the
importance weights are degenerate ⇒ distrust the estimate). Deterministic given a seed
(pure stdlib, no numpy).

Promotion rule (caller's, founder-gated): promote only when ci_low(lift) > 0 AND ess is
healthy. This module only MEASURES — it never promotes.

Reference: ainfera-vault methodology/dr-ope.md (Dudík et al. doubly-robust OPE).
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class LoggedSample:
    task_type: str  # context x
    action: str  # logged action a_i (chosen candidate)
    reward: float  # observed reward r_i in [0,1]
    propensity: float  # μ(a_i | x_i): prob the LOGGING policy chose a_i (in (0,1])


@dataclass(frozen=True)
class OPEResult:
    v_dr: float  # doubly-robust value estimate of the TARGET policy
    v_logged: float  # on-policy mean reward of the logging policy (baseline)
    lift: float  # v_dr − v_logged
    ci_low: float  # paired-bootstrap CI on the lift
    ci_high: float
    n: int  # samples used
    ess: float  # effective sample size of the importance weights
    mean_weight: float  # mean ρ (≈1 when target ≈ logging)


def _direct_value(
    task_type: str,
    target_probs: dict[str, dict[str, float]],
    q_hat: dict[tuple[str, str], float],
) -> float:
    """Σ_a π(a|x)·q̂(x,a) — the direct-method value at this context under the target."""
    probs = target_probs.get(task_type, {})
    return sum(p * q_hat.get((task_type, a), 0.0) for a, p in probs.items())


def _ess(weights: list[float]) -> float:
    """Kish effective sample size: (Σw)² / Σw². n means weights are uniform; ≪n means
    a few samples dominate ⇒ the IPS correction is unreliable."""
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    return 0.0 if s2 <= 0 else (s1 * s1) / s2


def _bootstrap_ci(
    values: list[float], alpha: float, draws: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean of `values`. Deterministic given seed."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(draws):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[min(draws - 1, int((alpha / 2) * draws))]
    hi = means[min(draws - 1, int((1 - alpha / 2) * draws))]
    return (lo, hi)


def evaluate_policy(
    samples: list[LoggedSample],
    target_probs: dict[str, dict[str, float]],
    q_hat: dict[tuple[str, str], float],
    *,
    weight_clip: float | None = 10.0,
    ci_alpha: float = 0.05,
    bootstrap: int = 2000,
    seed: int = 0,
) -> OPEResult:
    """Doubly-robust value of the target policy (given by `target_probs` + `q_hat`) on
    logged `samples`, vs the on-policy logged mean. CI is on the lift (paired bootstrap).

    `target_probs[task_type][action]` = π(a|x) (e.g. D7 alloc_weight).
    `q_hat[(task_type, action)]`      = q̂(x,a) (e.g. D1 q_empirical).
    Samples with propensity ≤ 0 contribute weight 0 (IPS term dropped, direct kept).
    """
    if not samples:
        return OPEResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0)

    dr_scores: list[float] = []
    rewards: list[float] = []
    weights: list[float] = []
    for s in samples:
        pi = target_probs.get(s.task_type, {}).get(s.action, 0.0)
        rho = 0.0 if s.propensity <= 0 else pi / s.propensity
        if weight_clip is not None:
            rho = min(rho, weight_clip)
        direct = _direct_value(s.task_type, target_probs, q_hat)
        q_sa = q_hat.get((s.task_type, s.action), 0.0)
        dr_scores.append(direct + rho * (s.reward - q_sa))
        rewards.append(s.reward)
        weights.append(rho)

    n = len(samples)
    v_dr = sum(dr_scores) / n
    v_logged = sum(rewards) / n
    lift_samples = [d - r for d, r in zip(dr_scores, rewards)]
    ci_low, ci_high = _bootstrap_ci(lift_samples, ci_alpha, bootstrap, seed)
    return OPEResult(
        v_dr=v_dr,
        v_logged=v_logged,
        lift=v_dr - v_logged,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
        ess=_ess(weights),
        mean_weight=sum(weights) / n,
    )


__all__ = ["LoggedSample", "OPEResult", "evaluate_policy"]
