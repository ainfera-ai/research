"""D1 shrinkage posterior — pure unit tests + linucb_refit integration (AIN-542)."""

from __future__ import annotations

import pytest

from labs.linucb_refit import fit
from labs.shrinkage import shrinkage_posterior

REWARD = lambda r: r["reward"]  # noqa: E731


def _rows(candidate: str, rewards: list[float], task: str = "reasoning") -> list[dict]:
    return [
        {"task_type": task, "chosen_candidate": candidate, "reward": r} for r in rewards
    ]


# ── pure shrinkage ────────────────────────────────────────────────────────


def test_no_data_returns_prior() -> None:
    e = shrinkage_posterior(0.9, 0.0, 0, prior_strength=20)
    assert e.q_posterior == 0.9
    assert e.prior_weight == 1.0


def test_prior_strength_zero_is_raw_mean() -> None:
    e = shrinkage_posterior(0.9, 1.5, 3, prior_strength=0.0)  # mean 0.5, prior ignored
    assert e.q_posterior == 0.5
    assert e.prior_weight == 0.0


def test_known_blend_point() -> None:
    # (20*0.9 + 10) / (20 + 20) = 0.7 ; prior contributes half
    e = shrinkage_posterior(0.9, 10.0, 20, prior_strength=20)
    assert e.q_posterior == pytest.approx(0.7)
    assert e.prior_weight == pytest.approx(0.5)


def test_prior_weight_decays_monotonically() -> None:
    weights = [
        shrinkage_posterior(0.8, 0.0, n, prior_strength=20).prior_weight
        for n in (0, 5, 20, 100, 1000)
    ]
    assert weights == sorted(weights, reverse=True)
    assert weights[0] == 1.0
    assert weights[-1] < 0.05  # benchmark nearly gone at n=1000


def test_large_n_converges_to_empirical() -> None:
    e = shrinkage_posterior(0.9, 200.0, 1000, prior_strength=20)  # 1000 obs of 0.2
    assert e.q_posterior == pytest.approx(0.2, abs=0.02)


def test_clamped_to_unit_interval() -> None:
    assert shrinkage_posterior(1.5, 0.0, 0, prior_strength=20).q_posterior == 1.0
    assert shrinkage_posterior(-0.5, 0.0, 0, prior_strength=20).q_posterior == 0.0


def test_validation() -> None:
    with pytest.raises(ValueError):
        shrinkage_posterior(0.5, 0.0, -1, prior_strength=20)
    with pytest.raises(ValueError):
        shrinkage_posterior(0.5, 0.0, 1, prior_strength=-1)


# ── fit() integration ─────────────────────────────────────────────────────


def test_fit_default_is_v0_no_shrinkage() -> None:
    pol = fit(_rows("m", [0.0, 0.0]), seed=42, reward_fn=REWARD)
    cell = pol.cells[0]
    assert cell.q_empirical == 0.0  # raw mean
    assert cell.q_prior is None
    assert "q_prior" not in pol.to_json()  # v0 schema byte-clean


def test_fit_shrinks_low_n_toward_prior() -> None:
    # empirical 0.0 but only n=2 → posterior pulled up toward the 0.9 benchmark
    pol = fit(
        _rows("m", [0.0, 0.0]),
        seed=42,
        reward_fn=REWARD,
        priors={"m": 0.9},
        prior_strength=20,
    )
    cell = pol.cells[0]
    assert cell.q_empirical == pytest.approx(18 / 22, abs=1e-4)
    assert cell.q_prior == 0.9
    assert cell.prior_weight == pytest.approx(20 / 22, abs=1e-4)
    assert "q_prior" in pol.to_json()


def test_fit_high_n_decays_prior() -> None:
    # n=200, empirical 0.0 → benchmark mostly washed out
    pol = fit(
        _rows("m", [0.0] * 200),
        seed=42,
        reward_fn=REWARD,
        priors={"m": 0.9},
        prior_strength=20,
    )
    cell = pol.cells[0]
    assert cell.q_empirical == pytest.approx(18 / 220, abs=1e-4)
    assert cell.prior_weight < 0.10


def test_fit_shrinkage_deterministic() -> None:
    rows = _rows("m", [0.3, 0.7, 0.5])
    a = fit(rows, seed=7, reward_fn=REWARD, priors={"m": 0.8}, prior_strength=20)
    b = fit(rows, seed=7, reward_fn=REWARD, priors={"m": 0.8}, prior_strength=20)
    assert a.to_json() == b.to_json()


def test_fit_candidate_without_prior_uses_raw_mean() -> None:
    # prior_strength>0 but no prior for "m" → raw mean, no audit fields (conservative)
    pol = fit(
        _rows("m", [0.0, 0.0]),
        seed=42,
        reward_fn=REWARD,
        priors={"other": 0.9},
        prior_strength=20,
    )
    cell = pol.cells[0]
    assert cell.q_empirical == 0.0
    assert cell.q_prior is None
