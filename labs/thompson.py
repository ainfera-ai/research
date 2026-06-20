"""thompson.py — Thompson-sampling exploration allocation with a min-sample floor (D7, AIN-542).

Cheapest-wins (+ a flat ε-floor) starves under-sampled models: a model that is never
sampled keeps its prior posterior forever and is never chosen — so it can never earn
its way in, even if it would be great. Thompson sampling fixes this by allocating
exploration in proportion to each candidate's POSTERIOR PROBABILITY OF BEING BEST:
draw q ~ Beta(α, β) per candidate; the one that samples highest wins that draw; the
fraction of draws a candidate wins is its allocation. High-uncertainty (low-n)
candidates win often enough to get sampled, and allocation sharpens to exploitation as
the posteriors narrow.

Built on the D1 shrinkage posterior — the same Beta(s·prior+Σr, s·(1-prior)+(n-Σr)).
With prior_strength=0 it reduces to Beta(Σr, n-Σr), i.e. Thompson on the raw empirical
counts.

A min-sample floor guarantees every eligible candidate ≥ floor_pct allocation until it
has at least `min_samples` observations, so nothing is starved to zero before a fair
trial.

Deterministic given a seed (CRN-safe): the Monte-Carlo draw uses a seeded
``random.Random`` — pure stdlib, no numpy.

Reference: ainfera-vault methodology/exploration-allocation.md (Thompson + min-sample floor).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

_EPS = 1e-9


@dataclass(frozen=True)
class Posterior:
    candidate: str
    alpha: float  # Beta α = prior_strength·prior + Σreward
    beta: float  # Beta β = prior_strength·(1-prior) + (n - Σreward)
    n: int  # observation count behind the posterior


def beta_posterior(
    prior: float, reward_sum: float, n: int, *, prior_strength: float
) -> tuple[float, float]:
    """Beta(α, β) parameters for a quality in [0,1]. α/(α+β) is the D1 shrinkage mean.

    Clamped to a tiny epsilon so the Beta is always proper (α, β > 0) even at n=0 with
    prior_strength=0 or a degenerate prior at 0/1.
    """
    alpha = prior_strength * prior + reward_sum
    beta = prior_strength * (1.0 - prior) + (n - reward_sum)
    return (max(alpha, _EPS), max(beta, _EPS))


def thompson_probabilities(
    posteriors: list[Posterior], *, draws: int = 4000, seed: int = 0
) -> dict[str, float]:
    """P[candidate is the argmax] under its Beta posterior, by Monte-Carlo (seeded).

    Returns a probability per candidate, summing to 1.0 (the natural Thompson
    allocation). Deterministic given (posteriors, draws, seed).
    """
    if not posteriors:
        return {}
    if len(posteriors) == 1:
        return {posteriors[0].candidate: 1.0}
    rng = random.Random(seed)
    wins = {p.candidate: 0 for p in posteriors}
    for _ in range(draws):
        best_c: str | None = None
        best_s = -1.0
        for p in posteriors:
            s = rng.betavariate(p.alpha, p.beta)
            if s > best_s:
                best_s, best_c = s, p.candidate
        assert best_c is not None
        wins[best_c] += 1
    return {c: w / draws for c, w in wins.items()}


def allocate_with_floor(
    probs: dict[str, float],
    n_by_candidate: dict[str, int],
    *,
    min_samples: int,
    floor_pct: float,
) -> dict[str, float]:
    """Blend Thompson probabilities with a min-sample floor.

    Every candidate with n < min_samples is guaranteed at least `floor_pct` of the
    allocation; the remaining budget is split among the rest by Thompson probability.
    The result is a distribution (sums to 1.0). If the floor can't be honored for all
    under-sampled candidates (len(under)·floor_pct ≥ 1), the whole budget is split
    evenly among them.
    """
    cands = list(probs)
    if not cands:
        return {}
    under = {c for c in cands if n_by_candidate.get(c, 0) < min_samples}
    reserved = len(under) * floor_pct
    if under and reserved >= 1.0:
        share = 1.0 / len(under)
        return {c: (share if c in under else 0.0) for c in cands}

    alloc = {c: (floor_pct if c in under else 0.0) for c in cands}
    remaining = 1.0 - reserved
    # The over-sampled set shares `remaining` by Thompson prob; if everyone is
    # under-sampled, the under set shares it (on top of their floor).
    pool = [c for c in cands if c not in under] or cands
    pool_mass = sum(probs[c] for c in pool)
    for c in pool:
        if pool_mass > _EPS:
            alloc[c] += remaining * probs[c] / pool_mass
        else:
            alloc[c] += remaining / len(pool)
    return alloc


__all__ = [
    "Posterior",
    "allocate_with_floor",
    "beta_posterior",
    "thompson_probabilities",
]
