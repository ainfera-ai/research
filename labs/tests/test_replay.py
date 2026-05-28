"""Tests for labs.replay_gate. Verifies the 4 PROMOTE/HOLD guards from L14.2."""

from __future__ import annotations

import pytest

from labs.replay_gate import (
    EXPLORATION_FLOOR_PCT,
    MAX_CELL_REGRESS_PCT,
    MIN_DELTA_PCT,
    MIN_SAMPLE_PER_CELL,
    decide,
)


def _cell(task_type: str, candidate: str, dnc: float, n: int = 100, explore: float = 0.10):
    return {
        "task_type": task_type,
        "candidate": candidate,
        "done_and_cheaper_pct": dnc,
        "n_held_out": n,
        "explore_pct": explore,
    }


def test_promote_when_all_guards_pass():
    """Clean +1.0pp gain across 2 cells, no regression, ample samples → PROMOTE."""
    incumbent = [_cell("code", "opus", 60.0), _cell("chat", "gpt", 55.0)]
    candidate = [_cell("code", "opus", 61.0), _cell("chat", "gpt", 56.0)]
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    assert v.decision == "PROMOTE"
    assert v.guard_delta_met
    assert v.guard_no_regression
    assert v.guard_exploration_floor
    assert v.guard_min_sample
    assert v.halted_reason is None


def test_hold_on_insufficient_delta():
    """+0.3pp < MIN_DELTA_PCT (0.5) → HOLD with delta_below_floor reason."""
    incumbent = [_cell("code", "opus", 60.0)]
    candidate = [_cell("code", "opus", 60.3)]
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    assert v.decision == "HOLD"
    assert not v.guard_delta_met
    assert "delta_below_floor" in v.halted_reason


def test_hold_on_cell_regression():
    """Big overall gain but one cell drops -3pp → HOLD."""
    incumbent = [
        _cell("code", "opus", 60.0),
        _cell("chat", "gpt", 70.0),
    ]
    candidate = [
        _cell("code", "opus", 80.0),
        _cell("chat", "gpt", 66.0),   # -4pp regression
    ]
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    assert v.decision == "HOLD"
    assert not v.guard_no_regression
    assert "regression" in v.halted_reason


def test_hold_on_exploration_floor_violation():
    """Cell falls below 5% exploration → HOLD."""
    incumbent = [_cell("code", "opus", 60.0, explore=0.10)]
    candidate = [_cell("code", "opus", 70.0, explore=0.02)]  # below 0.05
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    assert v.decision == "HOLD"
    assert not v.guard_exploration_floor


def test_hold_on_undersize_cell():
    """Cell has <30 rows → HOLD."""
    incumbent = [_cell("code", "opus", 60.0, n=25)]
    candidate = [_cell("code", "opus", 70.0, n=25)]
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    assert v.decision == "HOLD"
    assert not v.guard_min_sample


def test_thresholds_frozen():
    """Discipline #12 — these constants are moat lock. Changing requires
    founder + Tulkas co-sign. This test fails if they drift."""
    assert MIN_DELTA_PCT == 0.5
    assert MAX_CELL_REGRESS_PCT == -2.0
    assert EXPLORATION_FLOOR_PCT == 0.05
    assert MIN_SAMPLE_PER_CELL == 30


def test_verdict_serializes_to_json():
    """to_json() emits well-formed JSON with all guards + cells."""
    incumbent = [_cell("code", "opus", 60.0)]
    candidate = [_cell("code", "opus", 61.0)]
    v = decide(
        incumbent_cells=incumbent, candidate_cells=candidate,
        incumbent_version="v1", candidate_version="v2",
    )
    import json
    payload = json.loads(v.to_json())
    assert payload["decision"] == "PROMOTE"
    assert payload["guards"]["delta_met"] is True
    assert isinstance(payload["cells"], list)
    assert payload["cells"][0]["task_type"] == "code"
