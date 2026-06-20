"""Tests for the Stage-2 live wiring (AIN-459). No live calls — a FakeGateway
stands in for the Ainfera SDK seam, so these run in CI with no SDK + no network.
"""

from __future__ import annotations

import json

import pytest

from labs.eval_harness import arms, config, snapshot, taskset
from labs.eval_harness.cycle import run_cycle
from labs.eval_harness.gateway import GatewayResult
from labs.eval_harness.judge import GatewayJudge
from labs.eval_harness.live_arms import PanelArm, PinnedArm, RouterArm, build_live_arm
from labs.eval_harness.loader import freeze_from_rows, is_heldout


# --- fakes -----------------------------------------------------------------


def _cost_for(model: str) -> float:
    if model == config.ROUTED_MODEL:
        return 0.02  # arm C (router) — cheap per call
    if model == config.FUSION_SYNTHESIZER:
        return 0.02  # arm D synthesizer
    if "premium" in model:
        return 0.20  # arm A
    if "cheap" in model:
        return 0.01  # arm B
    if "panel" in model:
        return 0.03  # each arm D member
    return 0.05


class FakeGateway:
    """Records calls; returns canned GatewayResults. The judge model returns a
    score string; arms return an answer + a model-specific cost. No live calls."""

    def __init__(self, *, judge_score: str = "4") -> None:
        self.calls: list[dict] = []
        self.judge_score = judge_score

    def call(
        self, *, model, messages, max_tokens=None, task_type=None, routing_hint=None
    ):
        self.calls.append({"model": model, "task_type": task_type})
        if model == config.JUDGE_MODEL:
            return GatewayResult(self.judge_score, 0.0, 1, 1, model)
        return GatewayResult(f"answer from {model}", _cost_for(model), 10, 20, model)


def _taskset():
    rows = [
        {"id": "code-1", "task_type": "code", "prompt": "p1"},
        {"id": "code-2", "task_type": "code", "prompt": "p2"},
        {"id": "summ-1", "task_type": "summarize", "prompt": "p3"},
    ]
    return taskset.freeze(rows, version="live-test-v1")


SPECS = {s.arm: s for s in arms.arm_specs()}


# --- judge -----------------------------------------------------------------


def test_gateway_judge_parses_score():
    j = GatewayJudge(FakeGateway(judge_score="4"))
    lab = j.label(task={"id": "t1", "prompt": "p"}, arm="C", output="ans")
    assert lab.score == 4.0 and lab.success is True


def test_gateway_judge_unparseable_raises():
    j = GatewayJudge(FakeGateway(judge_score="no number here"))
    with pytest.raises(ValueError, match="parseable"):
        j.label(task={"id": "t1", "prompt": "p"}, arm="C", output="ans")


def test_gateway_judge_held_out():
    # The judge model must never be one of the arms (L3) — construction asserts it.
    GatewayJudge(FakeGateway())  # must not raise


# --- arms ------------------------------------------------------------------


def test_pinned_arm_cost():
    out = PinnedArm(SPECS["A"], FakeGateway()).produce(
        {"prompt": "p", "task_type": "code"}
    )
    assert out.routed is False and out.cost_usd == pytest.approx(0.20)


def test_router_arm_routes():
    out = RouterArm(SPECS["C"], FakeGateway()).produce(
        {"prompt": "p", "task_type": "code"}
    )
    assert out.routed is True and out.model_used == config.ROUTED_MODEL


def test_panel_arm_sums_member_and_synth_costs():
    out = PanelArm(SPECS["D"], FakeGateway()).produce(
        {"prompt": "p", "task_type": "code"}
    )
    # 3 panel members @0.03 + synth @0.02
    assert out.cost_usd == pytest.approx(0.03 * 3 + 0.02)


def test_build_live_arm_shapes():
    assert isinstance(build_live_arm(SPECS["A"], FakeGateway()), PinnedArm)
    assert isinstance(build_live_arm(SPECS["C"], FakeGateway()), RouterArm)
    assert isinstance(build_live_arm(SPECS["D"], FakeGateway()), PanelArm)


# --- cycle: refusal + full run --------------------------------------------


def test_cycle_refused_when_inert(monkeypatch):
    monkeypatch.delenv(config.LIVE_ENV, raising=False)
    with pytest.raises(RuntimeError, match="INERT"):
        run_cycle(
            _taskset(), generated_at="2026-06-15T00:00:00Z"
        )  # gateway=None → real path gated


def test_cycle_runs_with_fake_gateway(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    res = run_cycle(
        _taskset(), generated_at="2026-06-15T00:00:00Z", gateway=FakeGateway()
    )
    assert res.status == "completed"
    assert res.arms == ("A", "B", "C", "D")
    # 3 tasks × 4 arms = 12 ArmCalls
    assert res.n_calls == 12
    # All arms succeed (judge=4); C cheapest per success → win.
    assert res.win is True
    assert res.drift_types == ()
    # artifacts written to the private dir
    assert (tmp_path).exists()
    web = json.loads((tmp_path / "proof-snapshot-live-test-v1.json").read_text())
    assert {a["id"] for a in web["arms"]} == {"A", "B", "C", "D"}
    assert web["state"] == "measured"


def test_cycle_cost_cap_halts(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    monkeypatch.setattr(config, "COST_CAP_USD", 0.05)  # halts almost immediately
    res = run_cycle(
        _taskset(), generated_at="2026-06-15T00:00:00Z", gateway=FakeGateway()
    )
    assert res.status == "halted_cost_cap"
    assert res.cost_usd >= 0.05


def test_cycle_calls_are_excluded(monkeypatch, tmp_path):
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    # If any ArmCall were not Labs-excluded, run_cycle's assert_excluded would raise.
    res = run_cycle(
        _taskset(), generated_at="2026-06-15T00:00:00Z", gateway=FakeGateway()
    )
    assert res.n_calls == 12  # reached the end → every call passed assert_excluded


# --- snapshot --------------------------------------------------------------


def test_web_snapshot_shape_matches_fixture():
    from labs.eval_harness import metrics
    from labs.eval_harness.arms import ArmCall

    calls = []
    for tt in ("code", "summarize"):
        for arm, ok, c in [
            ("A", True, 0.2),
            ("B", False, 0.01),
            ("C", True, 0.02),
            ("D", True, 0.08),
        ]:
            calls.append(
                ArmCall(
                    arm,
                    f"{arm}-{tt}",
                    tt,
                    ok,
                    c,
                    config.LABS_TENANT,
                    True,
                    config.EVAL_RUN_TAG,
                )
            )
    web = snapshot.build_web_snapshot(
        version="v1",
        generated_at="2026-06-15T00:00:00Z",
        arm_metrics=metrics.compute_all(calls),
        type_table=metrics.by_arm_type(calls),
    )
    assert set(web) == {
        "version",
        "generated_at",
        "state",
        "source",
        "arms",
        "task_types",
    }
    assert set(web["arms"][0]) == {
        "id",
        "label",
        "kind",
        "cost_per_success",
        "success_rate",
    }
    assert web["task_types"][0]["task_type"] in {"code", "summarize"}
    snapshot.assert_sanitized(web)  # must not raise


def test_assert_sanitized_rejects_leak():
    bad = {
        "state": "measured",
        "arms": [
            {
                "id": "C",
                "label": "Ainfera",
                "kind": "router",
                "cost_per_success": 0.02,
                "success_rate": 0.9,
                "model_used": "claude-opus-4-8",
            }
        ],
    }
    with pytest.raises(AssertionError, match="non-sanitized"):
        snapshot.assert_sanitized(bad)


# --- loader ----------------------------------------------------------------


def test_loader_holdout_deterministic():
    assert is_heldout("abc") == is_heldout("abc")


def test_loader_freezes_heldout_subset():
    rows = [{"id": f"r{i}", "task_type": "code", "prompt": f"p{i}"} for i in range(200)]
    fz = freeze_from_rows(rows, version="hv1", holdout_pct=0.5)
    # ~50% held out; deterministic, hashed, version-pinned
    assert 0 < fz.n < 200
    assert fz.version == "hv1" and len(fz.hash) == 64
