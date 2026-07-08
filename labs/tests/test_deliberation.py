"""AIN-542 Step 5 · Tier C dispersion→deliberation pipeline — tests."""

from __future__ import annotations

from dataclasses import dataclass

from labs.deliberation import (
    DEFAULT_DISPERSION_THRESHOLD,
    DeliberationOutcome,
    EscalationRecord,
    VerdictView,
    build_escalation_roster,
    deliberate,
    deliberate_batch,
    down_weight_multiplier,
    is_low_confidence,
)
from labs.council import Vote
from labs.council_seats import COUNCIL_SEATS, SPARK_SEAT_POOL, Seat


# ── lightweight verdict stub (conforms to the structural Protocol) ───────────


@dataclass(frozen=True)
class _V:
    """Minimal verdict for testing — has exactly the fields the pipeline reads."""

    item_id: str
    dispersion: float
    confidence: float
    label: Vote = Vote.A
    n_seats: int = 5
    n_families: int = 5


# ── is_low_confidence ────────────────────────────────────────────────────────


def test_is_low_confidence_threshold() -> None:
    assert not is_low_confidence(0.3, threshold=0.4)
    assert not is_low_confidence(0.4, threshold=0.4)  # at threshold → NOT low
    assert is_low_confidence(0.41, threshold=0.4)
    assert is_low_confidence(0.9, threshold=0.4)


# ── down_weight_multiplier ──────────────────────────────────────────────────


def test_down_weight_no_penalty_below_threshold() -> None:
    assert down_weight_multiplier(0.0) == 1.0
    assert down_weight_multiplier(0.3, threshold=0.4) == 1.0
    assert down_weight_multiplier(0.4, threshold=0.4) == 1.0  # boundary


def test_down_weight_decreases_with_dispersion() -> None:
    dw_low = down_weight_multiplier(0.5, threshold=0.4)
    dw_mid = down_weight_multiplier(0.7, threshold=0.4)
    dw_high = down_weight_multiplier(0.95, threshold=0.4)
    assert 1.0 > dw_low > dw_mid > dw_high
    assert dw_high >= 0.2  # floor


def test_down_weight_never_below_floor() -> None:
    assert abs(down_weight_multiplier(1.0, floor=0.3) - 0.3) < 1e-9
    assert down_weight_multiplier(0.99, floor=0.2) >= 0.2


# ── deliberate: certify path ────────────────────────────────────────────────


def test_certify_when_dispersion_below_threshold() -> None:
    v = _V("i1", dispersion=0.2, confidence=0.9)
    out = deliberate(v, threshold=0.4)
    assert out.action == "certify"
    assert out.low_confidence is False
    assert out.down_weight == 1.0
    assert out.final_confidence == 0.9
    assert out.final_label == Vote.A
    assert out.history == []


# ── deliberate: down_weight path ────────────────────────────────────────────


def test_down_weight_when_no_rejudge() -> None:
    v = _V("i1", dispersion=0.8, confidence=0.9)
    out = deliberate(v, threshold=0.4)
    assert out.action == "down_weight"
    assert out.low_confidence is True
    assert out.down_weight < 1.0
    assert out.final_confidence < 0.9
    assert out.final_confidence == round(0.9 * out.down_weight, 4)
    assert out.history == []


# ── deliberate: escalate path (round-2 resolves) ────────────────────────────


def test_escalate_and_round2_resolves() -> None:
    v = _V("i1", dispersion=0.8, confidence=0.6)
    r2 = _V("i1", dispersion=0.2, confidence=0.95)

    def rejudge(item_id: str):
        assert item_id == "i1"
        return r2

    out = deliberate(v, threshold=0.4, rejudge=rejudge)
    assert out.action == "escalate"
    assert out.was_escalated
    assert len(out.history) == 1
    assert out.history[0].round_number == 1
    assert out.history[0].still_low_confidence is False
    assert out.final_confidence == 0.95  # round-2 verdict certified
    assert out.final_label == Vote.A
    assert out.down_weight == 1.0  # no down-weight on a resolved escalation


# ── deliberate: escalate path (round-2 still low → exhaust) ──────────────────


def test_escalate_exhausted_when_round2_still_low() -> None:
    v = _V("i1", dispersion=0.9, confidence=0.55)
    r2 = _V("i1", dispersion=0.85, confidence=0.5)

    def rejudge(item_id: str):
        return r2

    out = deliberate(v, threshold=0.4, rejudge=rejudge, max_rounds=1)
    assert out.action == "escalation_exhausted"
    assert out.was_escalated
    assert len(out.history) == 1
    assert out.history[0].still_low_confidence is True
    assert out.down_weight < 1.0
    assert out.final_confidence == round(0.5 * out.down_weight, 4)


# ── deliberate: multi-round escalation ──────────────────────────────────────


def test_multi_round_escalation_then_resolve() -> None:
    v = _V("i1", dispersion=0.9, confidence=0.5)
    r2 = _V("i1", dispersion=0.7, confidence=0.6)  # still low
    r3 = _V("i1", dispersion=0.1, confidence=0.95)  # resolved

    rounds = iter([r2, r3])

    def rejudge(item_id: str):
        return next(rounds)

    out = deliberate(v, threshold=0.4, rejudge=rejudge, max_rounds=2)
    assert out.action == "escalate"
    assert len(out.history) == 2
    assert out.history[0].round_number == 1
    assert out.history[0].still_low_confidence is True
    assert out.history[1].round_number == 2
    assert out.history[1].still_low_confidence is False
    assert out.final_confidence == 0.95


def test_multi_round_escalation_all_exhausted() -> None:
    v = _V("i1", dispersion=0.9, confidence=0.5)
    r2 = _V("i1", dispersion=0.8, confidence=0.55)
    r3 = _V("i1", dispersion=0.75, confidence=0.6)

    rounds = iter([r2, r3])

    def rejudge(item_id: str):
        return next(rounds)

    out = deliberate(v, threshold=0.4, rejudge=rejudge, max_rounds=2)
    assert out.action == "escalation_exhausted"
    assert len(out.history) == 2
    # final confidence = last round's confidence * down_weight on last dispersion
    dw = down_weight_multiplier(0.75, threshold=0.4)
    assert out.final_confidence == round(0.6 * dw, 4)


# ── deliberate: history is accumulated, not reset ──────────────────────────


def test_escalation_history_accumulates_rounds() -> None:
    v = _V("i1", dispersion=0.9, confidence=0.5)
    r2 = _V("i1", dispersion=0.9, confidence=0.5)

    def rejudge(item_id: str):
        return r2

    out = deliberate(v, threshold=0.4, rejudge=rejudge, max_rounds=2)
    assert len(out.history) == 2
    assert all(r.still_low_confidence for r in out.history)
    assert out.history[0].round_number == 1
    assert out.history[1].round_number == 2


# ── VerdictView adapter ─────────────────────────────────────────────────────


def test_verdict_view_from_council_verdict() -> None:
    from labs.council import CouncilVerdict

    cv = CouncilVerdict(
        item_id="x1",
        label=Vote.B,
        confidence=0.8,
        dispersion=0.3,
        n_seats=4,
        n_families=3,
        excluded_seats=["Námo"],
        seat_votes={"Manwë": Vote.B},
    )
    view = VerdictView.from_verdict(cv)
    assert view.item_id == "x1"
    assert view.dispersion == 0.3
    assert view.confidence == 0.8
    assert view.label == Vote.B
    assert view.n_seats == 4


def test_verdict_view_from_pointwise_verdict() -> None:
    from labs.council_pointwise import PointwiseVerdict

    pv = PointwiseVerdict(
        item_id="p1",
        consensus_score=3.5,
        passed=True,
        confidence=0.7,
        dispersion=0.5,
        n_seats=5,
        n_families=4,
        excluded_seats=[],
        seat_scores={"Námo": 4},
    )
    view = VerdictView.from_verdict(pv)
    assert view.item_id == "p1"
    assert view.dispersion == 0.5
    assert view.confidence == 0.7
    assert view.label is True  # falls back to `passed`
    assert view.n_seats == 5


# ── deliberate_batch ────────────────────────────────────────────────────────


def test_deliberate_batch_mixed_items() -> None:
    verdicts = [
        _V("low", dispersion=0.2, confidence=0.95),    # certify
        _V("high", dispersion=0.8, confidence=0.6),    # down_weight (no rejudge)
        _V("mid", dispersion=0.5, confidence=0.7),    # down_weight
    ]
    results = deliberate_batch(verdicts, threshold=0.4)
    assert len(results) == 3
    assert results[0].action == "certify"
    assert results[1].action == "down_weight"
    assert results[2].action == "down_weight"
    assert results[0].final_confidence == 0.95
    assert results[1].final_confidence < 0.6
    assert results[2].final_confidence < 0.7
    assert results[2].final_confidence > results[1].final_confidence  # less dispersion → less penalty


# ── build_escalation_roster ─────────────────────────────────────────────────


def test_build_escalation_roster_expand() -> None:
    expanded = build_escalation_roster(COUNCIL_SEATS, SPARK_SEAT_POOL, expand=True)
    assert len(expanded) == len(COUNCIL_SEATS) + len(SPARK_SEAT_POOL)
    # all original seats retained
    original_slugs = {s.model_slug for s in COUNCIL_SEATS}
    expanded_slugs = {s.model_slug for s in expanded}
    assert original_slugs.issubset(expanded_slugs)
    # no duplicates
    assert len(expanded_slugs) == len(expanded)


def test_build_escalation_roster_swap() -> None:
    swapped = build_escalation_roster(COUNCIL_SEATS, SPARK_SEAT_POOL, expand=False)
    assert len(swapped) == len(SPARK_SEAT_POOL)
    # entirely different panel
    original_slugs = {s.model_slug for s in COUNCIL_SEATS}
    swapped_slugs = {s.model_slug for s in swapped}
    assert original_slugs.isdisjoint(swapped_slugs)


# ── integration: real CouncilVerdict + deliberation ─────────────────────────


def test_deliberate_with_real_council_verdict() -> None:
    from labs.council import CouncilVerdict, Comparison, run_council

    comp = Comparison(
        "i1", "google", "meta",
        {"Námo": Vote.A, "Manwë": Vote.B, "Tulkas": Vote.TIE,
         "Aulë": Vote.A, "Yavanna": Vote.B, "Ulmo": Vote.A},
        [],
    )
    verdicts, _ = run_council([comp], COUNCIL_SEATS)
    v = verdicts[0]

    out = deliberate(v, threshold=0.4)
    # a split verdict should have high dispersion → either down_weight or escalate
    assert out.action in ("down_weight", "escalate", "escalation_exhausted")
    assert out.original_dispersion == v.dispersion
    assert out.original_confidence == v.confidence


# ── roster: Mistral seat + role designations ────────────────────────────────


def test_council_roster_has_mistral_seat() -> None:
    families = {s.family for s in COUNCIL_SEATS}
    assert "mistral" in families
    mistral_seats = [s for s in COUNCIL_SEATS if s.family == "mistral"]
    assert len(mistral_seats) == 1
    assert mistral_seats[0].persona == "Ulmo"
    assert mistral_seats[0].role == "member"


def test_council_roster_spans_six_disjoint_families() -> None:
    families = {s.family for s in COUNCIL_SEATS}
    assert len(families) >= 5  # constitution: ≥5 families
    expected = {"anthropic", "openai", "google", "xai", "meta", "mistral"}
    assert families == expected


def test_namo_is_chair() -> None:
    chair = [s for s in COUNCIL_SEATS if s.role == "chair"]
    assert len(chair) == 1
    assert chair[0].persona == "Námo"


def test_tulkas_is_dissent() -> None:
    dissent = [s for s in COUNCIL_SEATS if s.role == "dissent"]
    assert len(dissent) == 1
    assert dissent[0].persona == "Tulkas"


def test_no_duplicate_model_slugs_across_roster_and_pool() -> None:
    core_slugs = {s.model_slug for s in COUNCIL_SEATS}
    pool_slugs = {s.model_slug for s in SPARK_SEAT_POOL}
    assert core_slugs.isdisjoint(pool_slugs)


def test_pairwise_exclusion_still_leaves_floor_with_six_seats() -> None:
    import itertools
    from labs.council_seats import eligible_seats, families_of

    families = sorted({s.family for s in COUNCIL_SEATS})
    for fa, fb in itertools.combinations(families, 2):
        eligible, _ = eligible_seats({fa, fb})
        # 6 seats - ≤2 family exclusions → ≥4 seats / ≥3 families
        assert len(eligible) >= 4
        assert len(families_of(eligible)) >= 3
