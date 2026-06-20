"""Tests for labs.linucb_refit. CRN-deterministic given seed."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from labs.linucb_refit import fit


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def labeled_rows():
    return json.loads((FIXTURES / "synthetic_labeled.json").read_text())


def test_fit_deterministic_under_seed(labeled_rows):
    """Same labeled rows + same seed → byte-identical JSON output (CRN)."""
    today = datetime(2026, 6, 2, 20, 30, 0, tzinfo=timezone.utc)
    p1 = fit(labeled_rows, seed=20260528, today=today)
    p2 = fit(labeled_rows, seed=20260528, today=today)
    assert p1.to_json() == p2.to_json(), "LinUCB must be CRN-deterministic"


def test_fit_emits_one_cell_per_pair(labeled_rows):
    """Distinct (task_type, candidate) tuples → distinct cells."""
    pol = fit(labeled_rows, seed=42)
    cells = {(c.task_type, c.candidate) for c in pol.cells}
    assert len(cells) == len(pol.cells), "no duplicate cells"
    # fixture has 6 cell tuples
    assert len(cells) == 6


def test_q_empirical_in_unit_interval(labeled_rows):
    """q_empirical = (mean(judge_score) - 1) / 4 → ∈ [0, 1]."""
    pol = fit(labeled_rows, seed=42)
    for cell in pol.cells:
        assert 0.0 <= cell.q_empirical <= 1.0, f"cell {cell} q outside [0,1]"


def test_exploration_floor_enforced(labeled_rows):
    """Every cell.explore_pct ≥ exploration_floor_pct."""
    pol = fit(labeled_rows, seed=42, exploration_floor_pct=0.05)
    for cell in pol.cells:
        assert cell.explore_pct >= 0.05, f"cell {cell} below floor"


def test_alpha_affects_ucb(labeled_rows):
    """Higher alpha → wider UCB bonus → higher ucb."""
    low_alpha = fit(labeled_rows, seed=42, alpha=0.1)
    high_alpha = fit(labeled_rows, seed=42, alpha=2.0)
    # Compare same cell key in both
    low_by_key = {(c.task_type, c.candidate): c for c in low_alpha.cells}
    for hi_cell in high_alpha.cells:
        lo_cell = low_by_key[(hi_cell.task_type, hi_cell.candidate)]
        assert hi_cell.ucb >= lo_cell.ucb, "higher alpha should not decrease ucb"


def test_policy_version_format():
    """version: vYYYYMMDD-NNN."""
    today = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
    pol = fit([], seed=42, today=today)
    assert pol.version == "v20260615-001"
