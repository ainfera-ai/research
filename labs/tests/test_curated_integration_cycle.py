"""Tests for the curated SYNTHETIC integration cycle (AIN-459 / Tap-3).

No live calls — a FakeGateway (same pattern as test_eval_harness_live.py) stands
in for the Ainfera SDK seam, so these run in CI with no SDK + no network. They
prove the pipeline end-to-end on the committed curated task set and, critically,
that the emitted snapshot is tagged "illustrative" and can NEVER read as
"measured".
"""

from __future__ import annotations

from pathlib import Path

from labs.eval_harness import arms, config, snapshot, taskset
from labs.eval_harness.cycle import run_cycle
from labs.eval_harness.gateway import GatewayResult

CURATED_PATH = (
    Path(__file__).resolve().parents[1]
    / "eval_harness"
    / "fixtures"
    / "curated_integration_taskset.json"
)


# --- fake (mirrors test_eval_harness_live.py) ------------------------------


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
    """Records calls; returns canned GatewayResults. No live calls, no network."""

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


# --- curated task set -------------------------------------------------------


def test_curated_taskset_loads_without_hash_error():
    """The committed curated set loads via load_frozen (hash recomputes + matches)."""
    fz = taskset.load_frozen(CURATED_PATH)
    assert fz.version == "curated-synthetic-v20260616-001"
    assert len(fz.hash) == 64
    # 5 representative types, 20 tasks each → 100 tasks.
    assert fz.n == 100
    assert set(fz.by_type) == {"code", "summarize", "extract", "classify", "reasoning"}
    assert all(c == 20 for c in fz.by_type.values())
    # Every task carries the load contract shape.
    for t in fz.tasks:
        assert set(t) == {"id", "task_type", "prompt"}
        assert t["prompt"].strip()


def test_curated_taskset_is_undersized_for_stage2_power():
    """Documented: 20/type is below MIN_TASKS_PER_TYPE — expected for an
    integration cycle (the CLI warns, does not fail)."""
    fz = taskset.load_frozen(CURATED_PATH)
    undersized = taskset.assert_min_per_type(fz)
    assert set(undersized) == set(fz.by_type)  # all types are below 200/type


# --- cycle on the curated set: illustrative tagging (HONESTY-CRITICAL) ------


def test_curated_cycle_snapshot_is_illustrative_never_measured(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    fz = taskset.load_frozen(CURATED_PATH)
    res = run_cycle(
        fz,
        generated_at="2026-06-16T00:00:00Z",
        gateway=FakeGateway(),
        snapshot_state="illustrative",
        snapshot_source="SYNTHETIC integration cycle — pipeline proof only; NOT measured, NOT for publication.",
    )
    web = res.web_snapshot
    assert web is not None
    # The whole point: this snapshot is illustrative and must NEVER be measured.
    assert web["state"] == "illustrative"
    assert web["state"] != "measured"
    assert "SYNTHETIC" in web["source"]
    assert "NOT measured" in web["source"]
    # assert_sanitized still passes for an illustrative snapshot.
    snapshot.assert_sanitized(web)
    # And the written artifact on disk carries the illustrative tag too.
    import json

    on_disk = json.loads((tmp_path / f"proof-snapshot-{fz.version}.json").read_text())
    assert on_disk["state"] == "illustrative"
    assert on_disk["state"] != "measured"


def test_default_run_cycle_is_backward_compat_measured(tmp_path, monkeypatch):
    """No snapshot kwargs → unchanged behaviour: state == 'measured'."""
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    rows = [
        {"id": "code-1", "task_type": "code", "prompt": "p1"},
        {"id": "summ-1", "task_type": "summarize", "prompt": "p2"},
    ]
    fz = taskset.freeze(rows, version="bc-v1")
    res = run_cycle(fz, generated_at="2026-06-16T00:00:00Z", gateway=FakeGateway())
    assert res.web_snapshot is not None
    assert res.web_snapshot["state"] == "measured"
    # Default source is the genuine measured-run identity (unchanged).
    assert res.web_snapshot["source"] == snapshot.MEASURED_SOURCE
    snapshot.assert_sanitized(res.web_snapshot)


# --- pipeline math + L2 exclusion on every call -----------------------------


def test_curated_cycle_computes_metrics_and_excludes_every_call(tmp_path, monkeypatch):
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    # Raise the per-cycle cost cap so the full curated set runs end-to-end. (With
    # the default $25 cap and the fake's per-task cost the cycle would halt
    # part-way through — that hard cost ceiling is a real, working guardrail; we
    # lift it here only so the test exercises all 400 calls.)
    monkeypatch.setattr(config, "COST_CAP_USD", 1_000.0)
    fz = taskset.load_frozen(CURATED_PATH)
    res = run_cycle(
        fz,
        generated_at="2026-06-16T00:00:00Z",
        gateway=FakeGateway(),
        snapshot_state="illustrative",
        snapshot_source="SYNTHETIC integration cycle — pipeline proof only.",
    )
    assert res.status == "completed"
    assert res.arms == ("A", "B", "C", "D")
    # 100 tasks × 4 arms = 400 ArmCalls; reaching the end means run_cycle's
    # per-call arms.assert_excluded (L2) held on every one of them.
    assert res.n_calls == 400
    assert res.n_calls > 0
    # Metrics + win check computed (win is True here because the fake makes C
    # cheapest-per-success; the verdict is NOT meaningful on synthetic data).
    assert res.win is True
    assert res.drift_types == ()


def test_curated_cycle_cost_cap_halts_at_default(tmp_path, monkeypatch):
    """The real $25/cycle cost ceiling halts a 100-task synthetic cycle part-way
    (documents that a full curated run would need the cap raised / fewer tasks)."""
    monkeypatch.setenv(config.ARTIFACT_DIR_ENV, str(tmp_path))
    fz = taskset.load_frozen(CURATED_PATH)
    res = run_cycle(
        fz,
        generated_at="2026-06-16T00:00:00Z",
        gateway=FakeGateway(),
        snapshot_state="illustrative",
        snapshot_source="SYNTHETIC integration cycle — pipeline proof only.",
    )
    assert res.status == "halted_cost_cap"
    assert res.cost_usd >= config.COST_CAP_USD
    assert 0 < res.n_calls < 400
    # Even a halted cycle emits an illustrative (never measured) snapshot.
    assert res.web_snapshot is not None
    assert res.web_snapshot["state"] == "illustrative"


def test_every_armcall_passes_assert_excluded_directly():
    """Belt-and-suspenders L2: build the same calls and assert_excluded each one."""
    fz = taskset.load_frozen(CURATED_PATH)
    armset = tuple(s.arm for s in arms.arm_specs())
    n = 0
    for task in fz.tasks:
        for a in armset:
            call = arms.ArmCall(
                arm=a,
                task_id=task["id"],
                task_type=task["task_type"],
                success=True,
                cost_usd=0.01,
                tenant=config.LABS_TENANT,
                excluded_from_training=True,
                eval_run_tag=config.EVAL_RUN_TAG,
            )
            arms.assert_excluded(call)  # raises on any L2 violation
            n += 1
    assert n == 400
