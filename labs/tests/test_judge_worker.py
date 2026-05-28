"""Tests for labs.judge_worker. Deterministic (no DB, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.judge_worker import select_sample


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def unlabeled_rows():
    return json.loads((FIXTURES / "synthetic_unlabeled.json").read_text())


def test_select_sample_deterministic(unlabeled_rows):
    """Same inputs → same selected rows (CRN invariant)."""
    sel1 = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.20,
        cell_label_counts={},
    )
    sel2 = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.20,
        cell_label_counts={},
    )
    assert sel1 == sel2, "select_sample must be deterministic"


def test_cold_start_picks_more_when_cell_underlabeled(unlabeled_rows):
    """Cells with <10 labels use cold_start_pct (higher target)."""
    cold_selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.50,    # high cold-start
        cell_label_counts={},   # all cells in cold-start regime
    )
    warm_selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.50,
        # mark all cells fully warm — should use target_pct (0.05) only
        cell_label_counts={
            ("code", "claude-opus-4-7"): 100,
            ("code", "claude-sonnet-4-6"): 100,
            ("chat", "gpt-5-5"): 100,
            ("chat", "claude-sonnet-4-6"): 100,
            ("research", "deepseek-r1-32b"): 100,
            ("research", "claude-opus-4-7"): 100,
        },
    )
    assert len(cold_selected) >= len(warm_selected), (
        "cold-start should pick at least as many rows as warm at the same seed"
    )


def test_sample_max_caps_count(unlabeled_rows):
    """sample_max=2 returns at most 2 rows even with cold_start=100%."""
    selected = select_sample(
        unlabeled_rows=unlabeled_rows,
        sample_target_pct=1.0,
        sample_max=2,
        cold_start_pct=1.0,
        cell_label_counts={},
    )
    assert len(selected) <= 2


def test_empty_input_returns_empty():
    assert select_sample(
        unlabeled_rows=[],
        sample_target_pct=0.05,
        sample_max=100,
        cold_start_pct=0.20,
        cell_label_counts={},
    ) == []


def test_run_one_cycle_skeleton_raises():
    """W6 skeleton — real DB + Opus integration lands in AIN-290 follow-up."""
    from labs.judge_worker import run_one_cycle
    with pytest.raises(NotImplementedError):
        run_one_cycle()
