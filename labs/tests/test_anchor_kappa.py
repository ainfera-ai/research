"""AIN-542 Step 4 · anchor-κ calibration."""

from __future__ import annotations

from labs.anchor_kappa import (
    ANCHOR_KAPPA_GATE,
    ANCHOR_PAIRS_SQL,
    AnchorPair,
    build_pairs,
    cohen_kappa,
    compute_anchor_kappa,
    gwet_ac1,
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


def _competent(seat, task, first, second):
    # a competent judge agrees with verify(): prefers the GOOD output
    if "GOOD" in first and "GOOD" not in second:
        return "first"
    if "GOOD" in second and "GOOD" not in first:
        return "second"
    return "tie"


def _always_first(seat, task, first, second):
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


def test_gwet_ac1_bounds() -> None:
    assert gwet_ac1([]) is None
    perfect = [(Vote.A, Vote.A), (Vote.B, Vote.B), (Vote.A, Vote.A), (Vote.B, Vote.B)]
    assert gwet_ac1(perfect) == 1.0
    anti = [(Vote.A, Vote.B), (Vote.B, Vote.A), (Vote.A, Vote.B), (Vote.B, Vote.A)]
    assert gwet_ac1(anti) is not None and gwet_ac1(anti) < 0
    # cohen_kappa is now an alias for gwet_ac1 (constitution lock 2026-06-20)
    assert cohen_kappa(perfect) == gwet_ac1(perfect)
    assert cohen_kappa(anti) == gwet_ac1(anti)


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


# ── Step 4: seat reliability-filter (quarantine anchor-unreliable seats) ──────


def test_quarantine_is_directional() -> None:
    # DIRECTIONAL (scale-run fix): drop below-chance OR over-trusted seats; KEEP a
    # good seat that DS merely under-rates (high anchor accuracy, low DS).
    from labs.anchor_kappa import AnchorKappaResult, quarantine_seats

    r = AnchorKappaResult(
        kappa=1.0,
        n_pairs=10,
        council_accuracy=1.0,
        per_seat_anchor_accuracy={
            "good": 1.0,
            "underrated": 1.0,
            "belowchance": 0.3,
            "overconf": 0.55,
        },
        ds_reliability={
            "good": 0.95,
            "underrated": 0.05,
            "belowchance": 0.4,
            "overconf": 0.98,
        },
        divergence={},
        eligible=True,
    )
    bad = quarantine_seats(r, min_anchor_accuracy=0.5, max_overconfidence=0.4)
    assert "belowchance" in bad  # anchor accuracy 0.3 ≤ chance → bad
    assert "overconf" in bad  # DS 0.98 − anchor 0.55 = 0.43 > 0.4 → panel over-trusts
    assert "good" not in bad
    # the bug the scale run found: gemini was 100% anchor-accurate but DS-underrated,
    # and the old |DS−anchor| filter wrongly quarantined it. Must be KEPT now.
    assert "underrated" not in bad


def _seat(persona, slug):
    from labs.council_seats import Seat, family_of

    return Seat(persona, slug, family_of(slug), True, "seat")


def test_reliability_filter_quarantines_and_reaggregates() -> None:
    from labs.anchor_kappa import AnchorPair, compute_anchor_kappa

    roster = (
        _seat("G", "gemini-3-1-pro"),
        _seat("X", "grok-4"),
        _seat("M", "minimax-m3-novita"),
    )

    def caller(seat, task, first, second):
        gf = "GOOD" in first and "GOOD" not in second
        gs = "GOOD" in second and "GOOD" not in first
        if seat.persona == "M":  # the bad seat: always picks the WRONG one
            return "second" if gf else ("first" if gs else "tie")
        return "first" if gf else ("second" if gs else "tie")

    # candidate families (openai/deepseek) disjoint from the roster → no exclusion
    pairs = [
        AnchorPair(f"p{i}", "GOOD", "BAD", "openai", "deepseek", Vote.A, "t")
        if i % 2 == 0
        else AnchorPair(f"p{i}", "BAD", "GOOD", "deepseek", "openai", Vote.B, "t")
        for i in range(6)
    ]
    res = compute_anchor_kappa(pairs, caller, roster, reliability_filter=True)
    assert "M" in res.quarantined_seats  # the anti-correlated seat is quarantined
    assert (
        "M" not in res.per_seat_anchor_accuracy
    )  # and gone from the re-aggregated panel
    assert res.kappa == 1.0 and res.eligible  # filtered panel agrees with the anchor
