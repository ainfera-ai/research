"""Tests for labs.eval_harness — the proof-pipeline eval skeleton (AIN-458).

Stage 1 is INERT: flag-gated OFF, no live model calls. These tests pin the
guardrails (judge held out, Labs-only exclusion, flag default OFF), the metric
math, the Wilson CI, the task-set integrity hash, and the pre-registered win
check.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.eval_harness import arms, config, metrics, runner, taskset
from labs.eval_harness.arms import ArmCall, ArmSpec
from labs.eval_harness.judge import StubJudge, queue_for_human, success_from_score

FIXTURES = Path(__file__).parent / "fixtures"
TASKSET = FIXTURES / "eval_harness_taskset.json"


# --- helpers ---------------------------------------------------------------


def mk_calls(arm: str, task_type: str, *, n: int, successes: int, succ_cost: float,
             fail_cost: float = 0.001) -> list[ArmCall]:
    """n calls for one (arm, task_type); `successes` succeed at succ_cost each."""
    out: list[ArmCall] = []
    for i in range(n):
        ok = i < successes
        out.append(ArmCall(
            arm=arm, task_id=f"{arm}-{task_type}-{i}", task_type=task_type,
            success=ok, cost_usd=succ_cost if ok else fail_cost,
            tenant=config.LABS_TENANT, excluded_from_training=True,
            eval_run_tag=config.EVAL_RUN_TAG,
        ))
    return out


def winning_dataset() -> list[ArmCall]:
    """C matches A on success per type but is far cheaper than A and D."""
    calls: list[ArmCall] = []
    for tt in ("code", "summarize"):
        calls += mk_calls("A", tt, n=10, successes=9, succ_cost=0.20)
        calls += mk_calls("B", tt, n=10, successes=7, succ_cost=0.015)
        calls += mk_calls("C", tt, n=10, successes=9, succ_cost=0.02)
        calls += mk_calls("D", tt, n=10, successes=8, succ_cost=0.08)
    return calls


# --- flag gate -------------------------------------------------------------


def test_harness_disabled_by_default(monkeypatch):
    monkeypatch.delenv(config.ENABLED_ENV, raising=False)
    assert config.harness_enabled() is False
    res = runner.run_eval()
    assert res.status == "disabled"
    assert res.arms == ("A", "B", "C", "D")


def test_harness_enabled_dry_run(monkeypatch):
    monkeypatch.setenv(config.ENABLED_ENV, "true")
    res = runner.run_eval()
    assert res.status == "dry_run"
    assert res.judge_model == config.JUDGE_MODEL


def test_live_calls_refused_in_stage1(monkeypatch):
    monkeypatch.setenv(config.ENABLED_ENV, "true")
    with pytest.raises(RuntimeError, match="Stage 2"):
        runner.run_eval(live=True)


def test_stub_arm_makes_no_live_call():
    arm = arms.build_arms()["A"]
    with pytest.raises(NotImplementedError, match="Stage 2"):
        arm.run({"id": "x", "task_type": "code", "prompt": "noop"})


# --- L3: judge held out of all arms ----------------------------------------


def test_default_specs_hold_judge_out():
    arms.assert_judge_held_out()  # must not raise


def test_judge_leak_into_arm_fails_closed():
    bad = (ArmSpec("A", "premium_pin", (config.JUDGE_MODEL,)),)
    with pytest.raises(arms.JudgeLeakError):
        arms.assert_judge_held_out(bad)


def test_judge_leak_via_synthesizer_fails_closed():
    bad = (ArmSpec("D", "fusion_panel", ("m1", "m2"), synthesizer=config.JUDGE_MODEL),)
    with pytest.raises(arms.JudgeLeakError):
        arms.assert_judge_held_out(bad)


def test_stub_judge_pins_held_out_identity():
    j = StubJudge()
    assert j.model == config.JUDGE_MODEL
    with pytest.raises(NotImplementedError, match="Stage 2"):
        j.label(task={"id": "x"}, arm="C", output="...")


# --- L2: Labs-only exclusion -----------------------------------------------


def test_excluded_call_passes():
    [c] = mk_calls("C", "code", n=1, successes=1, succ_cost=0.02)
    arms.assert_excluded(c)  # must not raise


def test_wrong_tenant_fails_closed():
    c = ArmCall("C", "t", "code", True, 0.02, tenant="prod",
                excluded_from_training=True, eval_run_tag=config.EVAL_RUN_TAG)
    with pytest.raises(AssertionError, match="Labs"):
        arms.assert_excluded(c)


def test_not_excluded_from_training_fails_closed():
    c = ArmCall("C", "t", "code", True, 0.02, tenant=config.LABS_TENANT,
                excluded_from_training=False, eval_run_tag=config.EVAL_RUN_TAG)
    with pytest.raises(AssertionError, match="training"):
        arms.assert_excluded(c)


def test_validate_config_passes():
    runner.validate_config()  # default config is guardrail-clean


# --- metrics ---------------------------------------------------------------


def test_arm_metrics_basic():
    calls = mk_calls("C", "code", n=10, successes=9, succ_cost=0.02, fail_cost=0.001)
    m = metrics.compute_arm_metrics("C", calls)
    assert m.n == 10
    assert m.successes == 9
    assert m.success_rate == 0.9
    assert m.cost_per_success == pytest.approx(0.02)
    # cost_per_call = (9*0.02 + 1*0.001)/10
    assert m.cost_per_call == pytest.approx((9 * 0.02 + 0.001) / 10)


def test_cost_per_success_none_when_no_success():
    calls = mk_calls("B", "code", n=5, successes=0, succ_cost=0.0, fail_cost=0.01)
    m = metrics.compute_arm_metrics("B", calls)
    assert m.successes == 0
    assert m.cost_per_success is None


def test_floor_breaches_counted():
    calls = winning_dataset()
    # Drop C on summarize below A's floor.
    calls = [c for c in calls if not (c.arm == "C" and c.task_type == "summarize")]
    calls += mk_calls("C", "summarize", n=10, successes=6, succ_cost=0.02)
    allm = metrics.compute_all(calls)
    assert allm["C"].floor_breaches == 1  # only summarize trails the A floor


def test_floor_breaches_zero_when_at_floor():
    allm = metrics.compute_all(winning_dataset())
    assert allm["C"].floor_breaches == 0


# --- Wilson CI -------------------------------------------------------------


def test_wilson_ci_contains_point_estimate():
    lo, hi = metrics.wilson_ci(9, 10)
    assert 0.0 <= lo < 0.9 < hi <= 1.0


def test_wilson_ci_empty():
    assert metrics.wilson_ci(0, 0) == (0.0, 1.0)


def test_wilson_ci_clamped():
    lo, hi = metrics.wilson_ci(10, 10)
    assert hi == 1.0 and 0.0 < lo < 1.0


# --- task set: freeze / hash / version pin ---------------------------------


def test_taskset_loads_and_verifies_hash():
    fz = taskset.load_frozen(TASKSET)
    assert fz.version == "proof-eval-fixture-v20260615-001"
    assert fz.n == 10
    assert fz.by_type == {"code": 4, "extract": 3, "summarize": 3}


def test_taskset_hash_is_pinned():
    data = json.loads(TASKSET.read_text())
    assert taskset.canonical_hash(data["tasks"]) == data["hash"]


def test_taskset_hash_drift_fails_closed(tmp_path):
    data = json.loads(TASKSET.read_text())
    data["tasks"].append({"id": "sneaky", "task_type": "code", "prompt": "x"})
    p = tmp_path / "tampered.json"
    p.write_text(json.dumps(data))
    with pytest.raises(taskset.TaskSetIntegrityError):
        taskset.load_frozen(p)


def test_freeze_is_deterministic():
    data = json.loads(TASKSET.read_text())
    a = taskset.freeze(data["tasks"], version="v1")
    b = taskset.freeze(data["tasks"], version="v1")
    assert a.hash == b.hash


def test_fixture_is_undersized_for_stage2():
    fz = taskset.load_frozen(TASKSET)
    # Stage-1 fixture is intentionally < MIN_TASKS_PER_TYPE on every type.
    assert set(taskset.assert_min_per_type(fz)) == {"code", "extract", "summarize"}


# --- win check -------------------------------------------------------------


def test_win_check_c_wins():
    v = metrics.win_check(winning_dataset())
    assert v.win is True
    assert v.success_floor_met and v.cheaper_than_premium and v.cheaper_than_panel
    assert v.drift_types == ()


def test_win_check_drift_pages_founder():
    calls = [c for c in winning_dataset() if not (c.arm == "C" and c.task_type == "summarize")]
    calls += mk_calls("C", "summarize", n=10, successes=6, succ_cost=0.02)
    v = metrics.win_check(calls)
    assert v.win is False
    assert "summarize" in v.drift_types


def test_win_check_loses_when_not_cheaper():
    # C as expensive as A → fails the cost test even though success holds.
    calls = [c for c in winning_dataset() if c.arm != "C"]
    for tt in ("code", "summarize"):
        calls += mk_calls("C", tt, n=10, successes=9, succ_cost=0.20)
    v = metrics.win_check(calls)
    assert v.win is False
    assert v.cheaper_than_premium is False


# --- judge helpers ---------------------------------------------------------


def test_success_from_score_threshold():
    assert success_from_score(3.0) is True
    assert success_from_score(2.9) is False


def test_human_spotcheck_is_deterministic():
    a = queue_for_human("task-123")
    b = queue_for_human("task-123")
    assert a == b


def test_human_spotcheck_extremes():
    assert queue_for_human("x", pct=0.0) is False
    assert queue_for_human("x", pct=1.0) is True
