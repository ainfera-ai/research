"""AIN-542 Step 3 · Council aggregation — debias, Dawid–Skene, verdicts."""

from __future__ import annotations

from labs.council import (
    Vote,
    below_chance_seats,
    dawid_skene,
    make_comparison,
    needs_deliberation,
    position_debiased_vote,
    run_council,
)
from labs.council_seats import COUNCIL_SEATS


# ── position debiasing ───────────────────────────────────────────────────────


def test_consistent_pick_survives_both_orders() -> None:
    assert position_debiased_vote(Vote.A, Vote.A) is Vote.A
    assert position_debiased_vote(Vote.B, Vote.B) is Vote.B


def test_position_flip_collapses_to_tie() -> None:
    # disagreeing across orders = position bias / indifference → TIE
    assert position_debiased_vote(Vote.A, Vote.B) is Vote.TIE
    assert position_debiased_vote(Vote.A, Vote.TIE) is Vote.TIE


def test_make_comparison_cancels_a_pure_position_bias() -> None:
    # A seat that ALWAYS prefers whatever is shown first must net to TIE.
    def always_first(seat, first, second):
        return "first"

    comp = make_comparison("i1", "out_a", "out_b", "anthropic", "openai", always_first)
    assert set(comp.seat_votes.values()) == {Vote.TIE}
    # anthropic + openai seats excluded by family rule
    assert set(comp.excluded_seats) == {"Námo", "Manwë"}
    assert len(comp.seat_votes) == 3


def test_make_comparison_consistent_preference() -> None:
    def prefer_longer(seat, first, second):
        return "first" if len(first) >= len(second) else "second"

    comp = make_comparison(
        "i1", "a-longer-output", "short", "google", "meta", prefer_longer
    )
    # google + meta excluded → anthropic/openai/xai remain, all prefer A
    assert set(comp.excluded_seats) == {"Aulë", "Yavanna"}
    assert set(comp.seat_votes.values()) == {Vote.A}


# ── Dawid–Skene ──────────────────────────────────────────────────────────────


def _truth(i: int) -> Vote:
    return Vote.A if i % 2 == 0 else Vote.B


def _opposite(v: Vote) -> Vote:
    return Vote.B if v is Vote.A else Vote.A


def test_ds_recovers_labels_and_separates_reliable_from_adversarial() -> None:
    # 2 reliable seats vote truth; 1 adversarial seat votes opposite, every item.
    votes = {}
    for i in range(10):
        t = _truth(i)
        votes[f"it{i}"] = {"good1": t, "good2": t, "bad": _opposite(t)}
    ds = dawid_skene(votes)
    # latent labels track the reliable majority
    assert all(ds.labels[f"it{i}"] is _truth(i) for i in range(10))
    # reliable seats out-rank the adversarial one
    assert ds.reliability["good1"] > ds.reliability["bad"]
    assert ds.reliability["good2"] > ds.reliability["bad"]
    # the anti-correlated seat is at/below chance → flagged for filtering
    assert "bad" in below_chance_seats(ds.reliability)
    assert "good1" not in below_chance_seats(ds.reliability)


def test_ds_unanimous_is_high_confidence() -> None:
    votes = {f"it{i}": {"s1": Vote.A, "s2": Vote.A, "s3": Vote.A} for i in range(5)}
    ds = dawid_skene(votes)
    assert all(ds.labels[i] is Vote.A for i in votes)
    assert all(ds.label_probs[i][Vote.A] > 0.9 for i in votes)


def test_ds_handles_missing_votes() -> None:
    # incomplete annotation matrix (family-excluded seats absent on some items)
    votes = {
        "i1": {"s1": Vote.A, "s2": Vote.A},
        "i2": {"s2": Vote.B, "s3": Vote.B},
        "i3": {"s1": Vote.A, "s3": Vote.A},
    }
    ds = dawid_skene(votes)
    assert ds.labels["i1"] is Vote.A and ds.labels["i2"] is Vote.B


# ── verdict assembly ─────────────────────────────────────────────────────────


def _comp(item, votes, excluded):
    from labs.council import Comparison

    return Comparison(item, "google", "meta", votes, excluded)


def test_run_council_builds_verdict_records_with_floor_and_dispersion() -> None:
    comparisons = [
        _comp(
            "u1",
            {"Námo": Vote.A, "Manwë": Vote.A, "Tulkas": Vote.A},
            ["Aulë", "Yavanna"],
        ),
        _comp(
            "u2",
            {"Námo": Vote.A, "Manwë": Vote.B, "Tulkas": Vote.TIE},
            ["Aulë", "Yavanna"],
        ),
    ]
    verdicts, ds = run_council(comparisons)
    by_id = {v.item_id: v for v in verdicts}
    # unanimous u1: meets floor, low dispersion
    assert by_id["u1"].meets_floor
    assert by_id["u1"].n_seats == 3 and by_id["u1"].n_families == 3
    assert by_id["u1"].excluded_seats == ["Aulë", "Yavanna"]
    assert by_id["u1"].dispersion < by_id["u2"].dispersion
    # split u2: high dispersion → Tier C
    assert needs_deliberation(by_id["u2"]) or by_id["u2"].dispersion > 0.4


def test_verdict_floor_fails_below_three_seats() -> None:
    comparisons = [
        _comp("u1", {"Námo": Vote.A, "Manwë": Vote.A}, ["Aulë", "Yavanna", "Tulkas"])
    ]
    verdicts, _ = run_council(comparisons)
    assert verdicts[0].meets_floor is False  # only 2 seats


def test_live_roster_smoke() -> None:
    # the real roster drives make_comparison end-to-end with a fake caller
    def caller(seat, first, second):
        return "first"

    comp = make_comparison("x", "a", "b", "anthropic", "openai", caller, COUNCIL_SEATS)
    verdicts, _ = run_council([comp], COUNCIL_SEATS)
    assert verdicts[0].n_seats == 3
    assert isinstance(verdicts[0].label, Vote)


# ── AIN-546: degraded-roster visibility ──────────────────────────────────────


def test_verdict_records_unreachable_and_flags_degraded() -> None:
    comps = [
        _comp(
            "u1",
            {"Námo": Vote.A, "Manwë": Vote.A, "Tulkas": Vote.A},
            ["Aulë", "Yavanna"],
        )
    ]
    verdicts, _ = run_council(comps, unreachable_seats=("Vairë",))
    v = verdicts[0]
    assert v.unreachable_seats == ["Vairë"]
    assert v.degraded is True  # an unreachable seat → flagged, never silently certified
    # a full-floor verdict with no unreachable seats is NOT degraded
    assert run_council(comps)[0][0].degraded is False
