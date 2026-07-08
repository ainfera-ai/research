"""cuped.py — CUPED control-variate adjustment for variance reduction (AIN-558).

Constitution Part V Pillar 5 (Savings): "CUPED ≡ DR control variate". The same
variance-reduction idea that powers the doubly-robust estimator's direct-method term
appears here in its classical A/B-testing form: given a **pre-treatment covariate** X
(correlated with the outcome Y but unaffected by treatment assignment), adjust Y to

    Y_adj = Y − θ·(X − E[X])

where θ = Cov(X, Y) / Var(X).  The adjusted estimator is unbiased (E[Y_adj] = E[Y])
but has lower variance — Cov(Y, X)² / Var(X) is subtracted — so confidence intervals
tighten and the savings estimate stabilises faster.

In the savings pipeline (holdout.py), Y = CPST or incremental savings per request,
and X = a pre-period covariate (e.g. the same agent's historical CPST, or token count
on a matched prior task).  CUPED shrinks the CI on incremental_savings = CPST_control
− CPST_treatment, so billing-grade confidence arrives with fewer holdout samples.

Pure stdlib (math only).  No I/O.  Deterministic.

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data" — CUPED.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CupedPair:
    """One observation: pre-treatment covariate X and outcome Y for the same unit."""

    x: float  # pre-treatment covariate (must NOT be affected by treatment)
    y: float  # outcome (e.g. cost, reward, or per-request savings)


@dataclass(frozen=True)
class CupedResult:
    """CUPED-adjusted estimator result."""

    theta: float  # Cov(X,Y) / Var(X) — the optimal control-variate coefficient
    mean_x: float  # E[X] (sample mean of the covariate)
    mean_y: float  # E[Y] (unadjusted mean — identical to E[Y_adj])
    mean_y_adj: float  # E[Y_adj] = E[Y] (unbiased; only the variance changes)
    var_y: float  # Var(Y) — unadjusted sample variance
    var_y_adj: float  # Var(Y_adj) = Var(Y) − Cov(X,Y)² / Var(X)
    variance_reduction: float  # 1 − Var(Y_adj)/Var(Y)  (in [0, 1])
    n: int  # sample size
    correlation: float  # Pearson corr(X, Y) — 0 means no gain


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _variance(values: list[float], mu: float) -> float:
    """Population variance (1/n).  Used consistently for θ and variance_reduction."""
    n = len(values)
    if n == 0:
        return 0.0
    return sum((v - mu) ** 2 for v in values) / n


def _covariance(xs: list[float], ys: list[float], mx: float, my: float) -> float:
    """Population covariance (1/n)."""
    n = len(xs)
    if n == 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n


def compute_theta(pairs: list[CupedPair]) -> float:
    """Optimal CUPED coefficient θ = Cov(X, Y) / Var(X).

    Returns 0.0 when Var(X) = 0 (no covariate signal → no adjustment, Y_adj = Y).
    """
    if len(pairs) < 2:
        return 0.0
    xs = [p.x for p in pairs]
    ys = [p.y for p in pairs]
    mx = _mean(xs)
    my = _mean(ys)
    var_x = _variance(xs, mx)
    if var_x <= 0.0:
        return 0.0
    cov_xy = _covariance(xs, ys, mx, my)
    return cov_xy / var_x


def adjust(pairs: list[CupedPair]) -> CupedResult:
    """Compute the CUPED-adjusted estimator for a list of (X, Y) pairs.

    The adjusted values are Y_adj_i = Y_i − θ·(X_i − E[X]).
    The adjusted MEAN equals the unadjusted mean (unbiased). The adjusted VARIANCE
    is reduced by Cov(X,Y)² / Var(X) = ρ²·Var(Y), where ρ = corr(X,Y).

    Edge cases:
      · n < 2         → θ=0, no adjustment (insufficient data).
      · Var(X) = 0    → θ=0 (covariate is constant, no signal).
      · Var(Y) = 0    → variance_reduction=0 (nothing to reduce).
    """
    n = len(pairs)
    if n == 0:
        return CupedResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)

    xs = [p.x for p in pairs]
    ys = [p.y for p in pairs]
    mx = _mean(xs)
    my = _mean(ys)
    var_y = _variance(ys, my)
    theta = compute_theta(pairs)
    var_x = _variance(xs, mx)
    cov_xy = _covariance(xs, ys, mx, my)

    # Y_adj_i = Y_i - theta*(X_i - mean_x)
    adj_values = [y - theta * (x - mx) for x, y in zip(xs, ys)]
    mean_adj = _mean(adj_values)  # == my (unbiased)
    var_adj = _variance(adj_values, mean_adj)

    vr = 1.0 - (var_adj / var_y) if var_y > 0 else 0.0
    # Clamp: floating-point can push slightly above 0 when correlation ~ 0
    vr = max(0.0, min(1.0, vr))

    corr = cov_xy / (math.sqrt(var_x * var_y)) if var_x > 0 and var_y > 0 else 0.0

    return CupedResult(
        theta=theta,
        mean_x=mx,
        mean_y=my,
        mean_y_adj=mean_adj,
        var_y=var_y,
        var_y_adj=var_adj,
        variance_reduction=vr,
        n=n,
        correlation=corr,
    )


def adjusted_values(pairs: list[CupedPair]) -> list[float]:
    """Return the per-unit CUPED-adjusted Y values (Y_adj_i).

    Useful when the caller wants to feed adjusted outcomes into a downstream
    estimator (e.g. a bootstrap CI on incremental savings).
    """
    if not pairs:
        return []
    theta = compute_theta(pairs)
    mx = _mean([p.x for p in pairs])
    return [p.y - theta * (p.x - mx) for p in pairs]


__all__ = ["CupedPair", "CupedResult", "compute_theta", "adjust", "adjusted_values"]
