"""shrinkage.py — empirical-Bayes shrinkage posterior for per-cell quality (D1, AIN-542).

The router historically selected on a STATIC benchmark `q_prior` whose correlation
with realized reward is ~ -0.02 (i.e. noise). The daily refit then replaced it with
a RAW empirical mean — unbiased, but high-variance at low n (a cell with 2 labeled
rows could swing the router on luck).

The shrinkage posterior blends the two, weighting the data more as evidence accrues:

    q̂ = (prior_strength · prior + Σ reward) / (prior_strength + n)

  · n = 0          → q̂ = prior          (no data: fall back to the benchmark)
  · n → ∞          → q̂ → mean(reward)   (the benchmark decays fully away)
  · prior_strength = the prior's pseudo-count (how many observations it is worth)

`prior_weight = prior_strength / (prior_strength + n)` is the fraction of q̂ still
coming from the benchmark; it decays monotonically toward 0 as n grows — exactly the
"decay the benchmark prior as you calibrate on your own traffic" behaviour D1 wants.

This is the posterior mean of a Beta(prior_strength·prior, prior_strength·(1-prior))
prior updated by n reward observations in [0,1]. With `prior_strength = 0` it reduces
to the raw empirical mean (no shrinkage) — the v0 refit behaviour.

Reference: ainfera-vault methodology/q_empirical-v1.3.md (shrinkage extension).
"""

from __future__ import annotations

from dataclasses import dataclass

# The benchmark prior is worth this many observations before the data dominates.
# At n = prior_strength the posterior is a 50/50 blend; by n = 5·prior_strength the
# benchmark contributes < 17%. 20 is a deliberately weak prior (the benchmark is only
# ~ -0.02 correlated, so we want it to wash out fast once real reward arrives).
DEFAULT_PRIOR_STRENGTH = 20.0


@dataclass(frozen=True)
class ShrinkageEstimate:
    q_posterior: float  # the shrunk quality estimate, clamped to [0,1]
    prior_weight: (
        float  # fraction of q_posterior contributed by the prior (decays with n)
    )
    n: int  # observation count behind the estimate


def shrinkage_posterior(
    prior: float,
    reward_sum: float,
    n: int,
    *,
    prior_strength: float = DEFAULT_PRIOR_STRENGTH,
) -> ShrinkageEstimate:
    """Empirical-Bayes posterior mean blending `prior` with `n` observations.

    `prior` and each reward are assumed to lie in [0,1]; the result is clamped to
    [0,1] to stay a valid quality. `prior_strength >= 0` is the prior pseudo-count:
    0 → pure empirical mean (no shrinkage); larger → the benchmark holds longer.
    """
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")
    if n < 0:
        raise ValueError("n must be >= 0")
    denom = prior_strength + n
    if denom == 0:
        # no prior weight and no data — nothing to estimate; return the prior as-is.
        return ShrinkageEstimate(q_posterior=_clamp01(prior), prior_weight=1.0, n=0)
    q = (prior_strength * prior + reward_sum) / denom
    prior_weight = prior_strength / denom
    return ShrinkageEstimate(q_posterior=_clamp01(q), prior_weight=prior_weight, n=n)


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


__all__ = ["DEFAULT_PRIOR_STRENGTH", "ShrinkageEstimate", "shrinkage_posterior"]
