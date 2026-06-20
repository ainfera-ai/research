"""AIN-542 Step 4 · anchor-κ calibration."""

from __future__ import annotations

from labs.anchor_kappa import (
    ANCHOR_KAPPA_GATE,
    ANCHOR_PAIRS_SQL,
    AnchorPair,
    build_pairs,
    cohen_kappa,
    compute_anchor_kappa,
)
from labs.council import Vote


def _rows():
    # 6 prompts, each a verify=1 ("GOOD") output vs a verify=0 ("BAD") output,
    # from different families so family-exclusion leaves a quorum.
    fams = ["openai", "deepseek", "google", "minimax", "xai", "meta"]
    rows = []
    for i in range(6):
        gf, bf = fams[i % len(fams)], fams[(i + 1) % len(fams)]
        rows.append(
            {
                "prompt_key": f"p{i}",
                "output_text": f"GOOD-{i}",
                "family": gf,
                "verify_reward": 1.0,
            }
        )
        rows.append(
            {
                "prompt_key": f"p{i}",
                "output_text": f"BAD-{i}",
                "family": bf,
                "verify_reward": 0.0,
            }
        )
    return rows


def _competent(seat, first, second):
    # a competent judge agrees with verify(): prefers the GOOD output
    if "GOOD" in first and "GOOD" not in second:
        return "first"
    if "GOOD" in second and "GOOD" not in first:
        return "second"
    return "tie"


def _always_first(seat, first, second):
    return "first"


# ── pairing + κ ──────────────────────────────────────────────────────────────


def test_build_pairs_alternates_truth_side() -> None:
    pairs = build_pairs(_rows())
    assert len(pairs) == 6
    # winner side alternates so a position-biased Council can't game it
    truths = [p.truth for p in pairs]
    assert Vote.A in truths and Vote.B in truths
    # the GOOD output is always the truth side
    for p in pairs:
        winner = p.output_a if p.truth is Vote.A else p.output_b
        assert "GOOD" in winner


def test_cohen_kappa_bounds() -> None:
    assert cohen_kappa([]) is None
    perfect = [(Vote.A, Vote.A), (Vote.B, Vote.B), (Vote.A, Vote.A), (Vote.B, Vote.B)]
    assert cohen_kappa(perfect) == 1.0
    anti = [(Vote.A, Vote.B), (Vote.B, Vote.A), (Vote.A, Vote.B), (Vote.B, Vote.A)]
    assert cohen_kappa(anti) is not None and cohen_kappa(anti) < 0


# ── full calibration ─────────────────────────────────────────────────────────


def test_competent_council_clears_the_gate() -> None:
    res = compute_anchor_kappa(build_pairs(_rows()), _competent)
    assert res.kappa == 1.0
    assert res.eligible is True
    assert res.council_accuracy == 1.0
    assert res.n_pairs == 6
    # every seat that voted matched the anchor; DS≈anchor → low divergence
    assert all(a == 1.0 for a in res.per_seat_anchor_accuracy.values())
    assert res.max_divergence < 0.2


def test_position_biased_council_fails_the_gate() -> None:
    # a seat that always says "first" flips with order → debias → TIE everywhere,
    # so the Council reaches no decisive verdict and cannot clear the gate.
    res = compute_anchor_kappa(build_pairs(_rows()), _always_first)
    assert res.eligible is False
    assert res.council_accuracy == 0.0


def test_gate_constant_and_sql_shape() -> None:
    assert ANCHOR_KAPPA_GATE == 0.60
    assert "reward_source = 'verify'" in ANCHOR_PAIRS_SQL
    assert "NOT ro.exclude_from_training" in ANCHOR_PAIRS_SQL
    assert "prompt_key" in ANCHOR_PAIRS_SQL


def test_anchor_pair_dataclass() -> None:
    p = AnchorPair("i", "a", "b", "openai", "google", Vote.A)
    assert p.truth is Vote.A and p.family_a == "openai"
