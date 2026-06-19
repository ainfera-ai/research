"""AIN-542 Step 1 · pin the task-type verifiability map."""

from __future__ import annotations

from labs.task_verifiability import (
    CANONICAL_TASK_TYPES,
    TASK_VERIFIABILITY,
    Verifiability,
    is_verifiable,
    verifiability_of,
)


def test_map_covers_exactly_the_seven_canonical_types() -> None:
    # MIRROR api services.section16.VALID_TASK_TYPES — if these drift, fix both.
    assert CANONICAL_TASK_TYPES == frozenset(
        {"reasoning", "code", "extraction", "chat", "tool_use", "embed", "general"}
    )
    assert set(TASK_VERIFIABILITY) == CANONICAL_TASK_TYPES


def test_tiers_are_as_designed() -> None:
    assert verifiability_of("code").tier is Verifiability.VERIFIABLE
    assert verifiability_of("extraction").tier is Verifiability.VERIFIABLE
    assert verifiability_of("tool_use").tier is Verifiability.PARTIAL
    assert verifiability_of("reasoning").tier is Verifiability.PARTIAL
    for subj in ("chat", "embed", "general"):
        assert verifiability_of(subj).tier is Verifiability.SUBJECTIVE


def test_intrinsic_flag_marks_live_usable_checks() -> None:
    # code/extraction/tool_use have a no-gold intrinsic check; reasoning needs gold.
    assert verifiability_of("code").intrinsic is True
    assert verifiability_of("extraction").intrinsic is True
    assert verifiability_of("tool_use").intrinsic is True
    assert verifiability_of("reasoning").intrinsic is False


def test_unknown_and_null_default_to_subjective() -> None:
    assert verifiability_of(None).tier is Verifiability.SUBJECTIVE
    assert verifiability_of("not_a_task").tier is Verifiability.SUBJECTIVE
    assert is_verifiable(None) is False
    assert is_verifiable("code") is True
