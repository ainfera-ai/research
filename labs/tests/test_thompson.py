"""D7 Thompson allocation — pure unit tests + linucb_refit post-processor (AIN-542)."""

from __future__ import annotations

import pytest

from labs.linucb_refit import apply_thompson_allocation, fit
from labs.thompson import (
    Posterior,
    allocate_with_floor,
    beta_posterior,
    thompson_probabilities,
)

REWARD = lambda r: r["reward"]  # noqa: E731


def _rows(candidate: str, rewards: list[float], task: str = "reasoning") -> list[dict]:
    return [
        {"task_type": task, "chosen_candidate": candidate, "reward": r} for r in rewards
    ]


# ── beta_posterior ─────────────────────────────────────────────────────────


def test_beta_posterior_matches_shrinkage_mean() -> None:
    # prior 0.9, s=20, n=2, Σr=0 → α=18, β=20·0.1+2=4 ; mean 18/22 = D1 shrinkage mean
    a, b = beta_posterior(0.9, 0.0, 2, prior_strength=20)
    assert a == pytest.approx(18.0)
    assert b == pytest.approx(4.0)
    assert a / (a + b) == pytest.approx(18 / 22)


def test_beta_posterior_clamped_proper() -> None:
    a, b = beta_posterior(1.0, 0.0, 0, prior_strength=0.0)  # would be (0,0)
    assert a > 0 and b > 0


# ── thompson_probabilities ─────────────────────────────────────────────────


def test_probabilities_sum_to_one() -> None:
    post = [
        Posterior("a", 5, 5, 10),
        Posterior("b", 2, 8, 10),
        Posterior("c", 8, 2, 10),
    ]
    probs = thompson_probabilities(post, draws=2000, seed=1)
    assert sum(probs.values()) == pytest.approx(1.0)


def test_obvious_winner_dominates() -> None:
    post = [Posterior("good", 90, 10, 100), Posterior("bad", 10, 90, 100)]
    probs = thompson_probabilities(post, draws=4000, seed=1)
    assert probs["good"] > 0.99


def test_deterministic_under_seed() -> None:
    post = [Posterior("a", 5, 5, 10), Posterior("b", 6, 4, 10)]
    assert thompson_probabilities(post, draws=2000, seed=7) == thompson_probabilities(
        post, draws=2000, seed=7
    )


def test_single_candidate() -> None:
    assert thompson_probabilities([Posterior("solo", 3, 3, 6)]) == {"solo": 1.0}


def test_uncertain_candidate_still_explored() -> None:
    # same mean (0.5) but "new" is wide (n≈1) vs "settled" tight (n≈500): the wide one
    # must still win a meaningful share — that's the exploration Thompson buys.
    post = [Posterior("new", 1, 1, 1), Posterior("settled", 250, 250, 500)]
    probs = thompson_probabilities(post, draws=4000, seed=3)
    assert probs["new"] > 0.20


# ── allocate_with_floor ────────────────────────────────────────────────────


def test_floor_rescues_starved_candidate() -> None:
    # "starved" has ~0 posterior mass but n=1 < min_samples → floor guarantees it a share
    alloc = allocate_with_floor(
        {"good": 0.98, "starved": 0.02},
        {"good": 100, "starved": 1},
        min_samples=30,
        floor_pct=0.10,
    )
    assert alloc["starved"] >= 0.10
    assert alloc["good"] == pytest.approx(0.90)
    assert sum(alloc.values()) == pytest.approx(1.0)


def test_no_floor_when_all_sampled() -> None:
    alloc = allocate_with_floor(
        {"a": 0.7, "b": 0.3}, {"a": 50, "b": 50}, min_samples=30, floor_pct=0.10
    )
    assert alloc["a"] == pytest.approx(0.7)
    assert alloc["b"] == pytest.approx(0.3)


def test_floor_too_large_splits_evenly() -> None:
    alloc = allocate_with_floor(
        {"x": 0.9, "y": 0.1}, {"x": 0, "y": 0}, min_samples=30, floor_pct=0.6
    )  # 2·0.6 ≥ 1 → even split among the under-sampled
    assert alloc["x"] == pytest.approx(0.5)
    assert alloc["y"] == pytest.approx(0.5)


# ── apply_thompson_allocation (policy post-processor) ──────────────────────


def _policy_three():
    rows = (
        _rows("good", [0.9] * 50)
        + _rows("bad", [0.1] * 50)
        + _rows("newbad", [0.1, 0.1])  # n=2 < min_samples → starvation candidate
    )
    return fit(rows, seed=42, reward_fn=REWARD)


def test_apply_sets_alloc_and_sums_to_one() -> None:
    pol = apply_thompson_allocation(
        _policy_three(), min_samples=30, floor_pct=0.05, draws=2000
    )
    weights = [c.alloc_weight for c in pol.cells]
    assert all(w is not None for w in weights)
    assert sum(weights) == pytest.approx(1.0)  # single task_type → one distribution
    assert "alloc_weight" in pol.to_json()


def test_apply_floor_rescues_low_n_cell() -> None:
    pol = apply_thompson_allocation(
        _policy_three(), min_samples=30, floor_pct=0.05, draws=2000
    )
    by_cand = {c.candidate: c.alloc_weight for c in pol.cells}
    assert by_cand["newbad"] >= 0.05  # starved-but-young still gets a fair trial
    assert by_cand["good"] > by_cand["bad"]  # exploitation still favors the winner


def test_apply_is_deterministic() -> None:
    a = apply_thompson_allocation(_policy_three(), seed=11, draws=2000)
    b = apply_thompson_allocation(_policy_three(), seed=11, draws=2000)
    assert a.to_json() == b.to_json()


def test_apply_does_not_mutate_without_call() -> None:
    # a plain fitted policy carries no alloc_weight (v0 byte-clean)
    assert "alloc_weight" not in _policy_three().to_json()
