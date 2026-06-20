"""AIN-542 Step 3 · Council roster + self-preference family-exclusion."""

from __future__ import annotations

import itertools

from labs.council_seats import (
    COUNCIL_SEATS,
    eligible_seats,
    families_of,
)


def test_roster_spans_at_least_five_disjoint_families() -> None:
    families = {s.family for s in COUNCIL_SEATS}
    assert len(families) >= 5
    assert len({s.model_slug for s in COUNCIL_SEATS}) == len(COUNCIL_SEATS)


def test_family_exclusion_removes_both_candidate_families() -> None:
    eligible, excluded = eligible_seats({"anthropic", "openai"})
    ex = {s.family for s in excluded}
    assert ex == {"anthropic", "openai"}
    assert all(s.family not in {"anthropic", "openai"} for s in eligible)


def test_pairwise_exclusion_always_leaves_floor() -> None:
    # Any A-vs-B comparison excludes ≤2 families → ≥3 seats / ≥2 families remain.
    families = sorted({s.family for s in COUNCIL_SEATS})
    for fa, fb in itertools.combinations(families, 2):
        eligible, _ = eligible_seats({fa, fb})
        assert len(eligible) >= 3
        assert len(families_of(eligible)) >= 2


def test_same_family_candidates_exclude_one_family() -> None:
    eligible, excluded = eligible_seats({"anthropic"})
    assert {s.family for s in excluded} == {"anthropic"}
    assert len(eligible) == len(COUNCIL_SEATS) - 1


# ── AIN-546: canonical family_of + roster lockstep ───────────────────────────


def test_family_of_canonical_and_roster_lockstep() -> None:
    from labs.council_seats import family_of

    assert family_of("claude-opus-4-7") == "anthropic"
    assert family_of("gemini-3-1-pro") == "google"
    assert family_of("minimax-m3-novita") == "minimax"
    assert family_of("mystery-x") == "unknown"
    # every roster seat's declared family MUST equal the canonical map for its slug
    # (else self-preference family-exclusion silently fails to exclude that seat)
    for s in COUNCIL_SEATS:
        assert s.family == family_of(s.model_slug), s.persona
