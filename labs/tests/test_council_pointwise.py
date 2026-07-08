"""AIN-546 · Pointwise cross-family Council — family-exclusion, floor, consensus.

Deterministic, no live calls (fake PointwiseSeatCaller). Asserts the panel that
replaces the single-seat Super judge: self-preference firewall, the ≥3-seat/
≥2-family floor, consensus binarization on the anchor-κ axis, dispersion → Tier C,
and honest abstention (never a fabricated score)."""

from __future__ import annotations

from labs.council_pointwise import (
    COUNCIL_PASS_MIN_SCORE,
    PointwiseVerdict,
    needs_deliberation,
    parse_score,
    run_pointwise_council,
    score_response,
)
from labs.council_seats import COUNCIL_SEATS, family_of


def _caller(scores_by_persona):
    """Fake seat caller: persona -> score (or None to abstain)."""
    def call(seat, task, response):  # noqa: ARG001
        return scores_by_persona.get(seat.persona)
    return call


def _all(score):
    return _caller({s.persona: score for s in COUNCIL_SEATS})


# ── self-preference family-exclusion ─────────────────────────────────────────


def test_excludes_the_candidates_own_family() -> None:
    # response came from an anthropic model → Námo (claude/anthropic) must not score it.
    v = score_response("i1", "T", "ans", "anthropic", _all(4))
    assert "Námo" in v.excluded_seats
    assert "Námo" not in v.seat_scores
    # the remaining 5 seats span 5 distinct non-anthropic families.
    assert v.n_seats == 5
    assert v.n_families == 5
    assert "anthropic" not in {family_of_persona(p) for p in v.seat_scores}


def family_of_persona(persona: str) -> str:
    slug = {s.persona: s.model_slug for s in COUNCIL_SEATS}[persona]
    return family_of(slug)


def test_meets_floor_with_one_family_excluded() -> None:
    v = score_response("i1", "T", "ans", "openai", _all(4))
    assert v.meets_floor is True
    assert not v.degraded


# ── consensus + the anchor-κ binarization ────────────────────────────────────


def test_unanimous_pass() -> None:
    v = score_response("i1", "T", "ans", "google", _all(4))
    assert v.consensus_score == 4.0
    assert v.passed is True
    assert v.dispersion == 0.0
    assert v.confidence == 1.0


def test_unanimous_fail() -> None:
    v = score_response("i1", "T", "ans", "google", _all(2))
    assert v.consensus_score == 2.0
    assert v.passed is False


def test_threshold_is_the_anchor_axis() -> None:
    # consensus exactly at COUNCIL_PASS_MIN_SCORE passes (>= axis).
    v = score_response("i1", "T", "ans", "google", _all(COUNCIL_PASS_MIN_SCORE))
    assert v.passed is True


# ── dispersion → Tier C ──────────────────────────────────────────────────────


def test_split_panel_flags_deliberation() -> None:
    # exclude anthropic; the 5 eligible seats split 1/1/5/5/5.
    caller = _caller({"Manwë": 1, "Aulë": 1, "Tulkas": 5, "Yavanna": 5, "Ulmo": 5})
    v = score_response("i1", "T", "ans", "anthropic", caller)
    assert v.dispersion > 0.6
    assert needs_deliberation(v) is True
    assert v.consensus_score == 3.4  # mean of 1,1,5,5,5


def test_unanimous_does_not_need_deliberation() -> None:
    v = score_response("i1", "T", "ans", "google", _all(5))
    assert needs_deliberation(v) is False


# ── honest abstention (never a fabricated score) ─────────────────────────────


def test_abstaining_seat_is_unreachable_not_scored() -> None:
    # Yavanna abstains (None); excluded family = openai → Manwë excluded.
    caller = _caller({"Námo": 4, "Aulë": 4, "Tulkas": 4, "Yavanna": None, "Ulmo": 4})
    v = score_response("i1", "T", "ans", "openai", caller)
    assert "Yavanna" in v.unreachable_seats
    assert "Yavanna" not in v.seat_scores
    assert v.n_seats == 4  # Námo, Aulë, Tulkas, Ulmo scored; Manwë excluded (openai)
    assert v.degraded is True  # an unreachable seat makes the roster degraded


def test_whole_panel_abstains_yields_null_verdict() -> None:
    caller = _caller({})  # everyone abstains
    v = score_response("i1", "T", "ans", "google", caller)
    assert v.consensus_score is None
    assert v.passed is None
    assert v.meets_floor is False
    assert v.n_seats == 0


def test_floor_not_met_when_too_few_seats_score() -> None:
    # only 2 seats produce a score → below the 3-seat floor.
    caller = _caller({"Námo": 4, "Manwë": 4})
    v = score_response("i1", "T", "ans", "google", caller)
    assert v.n_seats == 2
    assert v.meets_floor is False
    assert v.degraded is True


# ── reliability weighting (Step-4 hook) ──────────────────────────────────────


def test_reliability_downweights_a_seat() -> None:
    caller = _caller({"Manwë": 1, "Aulë": 5, "Tulkas": 5, "Yavanna": 5, "Ulmo": 5})
    # uniform mean of 1,5,5,5,5 = 4.2; down-weighting the dissenter pulls it higher.
    base = score_response("i1", "T", "ans", "anthropic", caller)
    weighted = score_response(
        "i1", "T", "ans", "anthropic", caller,
        seat_reliability={"Manwë": 0.1, "Aulë": 1.0, "Tulkas": 1.0, "Yavanna": 1.0, "Ulmo": 1.0},
    )
    assert base.consensus_score == 4.2
    assert weighted.consensus_score > base.consensus_score


# ── parsing ──────────────────────────────────────────────────────────────────


def test_parse_score() -> None:
    assert parse_score("4") == 4
    assert parse_score("I'd say 5.") == 5
    assert parse_score("Score: 3/5") == 3
    assert parse_score("garbage") is None
    assert parse_score("") is None
    assert parse_score(None) is None


# ── batch ────────────────────────────────────────────────────────────────────


def test_batch_folds_in_unreachable_roster() -> None:
    items = [
        ("a", "T", "ans1", "google", ),
        ("b", "T", "ans2", "openai", ),
    ]
    verdicts = run_pointwise_council(
        [(i, t, r, f) for (i, t, r, f) in items],
        _all(4),
        unreachable_seats=("Estë",),
    )
    assert len(verdicts) == 2
    assert all("Estë" in v.unreachable_seats for v in verdicts)
    assert all(isinstance(v, PointwiseVerdict) for v in verdicts)
