"""cycle.py — one live eval cycle. INERT until the founder enables.

Ties the pieces together: frozen task set -> run arms A/B/C/D on each task ->
judge-label (held out) -> per-arm + per-type metrics + win check -> sanitized web
snapshot artifact + drift signal. A hard cost cap halts mid-cycle.

Refuses to build the real gateway unless ALL of:
  - LABS_EVAL_LIVE is on,
  - LABS_EVAL_PROBE_AGENT_ID is set (arm C exclusion — L2),
  - AINFERA_API_KEY is set (the Labs key).
Tests inject a fake gateway/judge to exercise the flow with no live calls.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from labs.eval_harness import arms, config, metrics
from labs.eval_harness.arms import ArmCall
from labs.eval_harness.gateway import GatewayClient
from labs.eval_harness.judge import GatewayJudge
from labs.eval_harness.live_arms import build_live_arm
from labs.eval_harness.snapshot import build_web_snapshot, write_artifacts
from labs.eval_harness.taskset import FrozenTaskSet

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CycleResult:
    status: str                       # "completed" | "halted_cost_cap"
    n_calls: int
    cost_usd: float
    arms: tuple[str, ...]
    win: bool | None
    drift_types: tuple[str, ...]       # non-empty → page founder (L: drift alert)
    web_snapshot: dict[str, Any] | None
    artifacts: dict[str, str] = field(default_factory=dict)
    note: str = ""


def _assert_live_allowed() -> None:
    if not config.live_enabled():
        raise RuntimeError(
            f"{config.LIVE_ENV} is OFF — the Stage-2 live cycle is INERT. Enable on "
            f"the Labs tenant only, after the Doppler key-fix + the 2026-06-19 gate."
        )
    if not os.environ.get(config.PROBE_AGENT_ID_ENV):
        raise RuntimeError(
            f"{config.PROBE_AGENT_ID_ENV} unset — arm C runs as a probe agent so its "
            f"routed calls are excluded from routing_outcomes (L2). Configure it first."
        )
    if not os.environ.get(config.GATEWAY_KEY_ENV):
        raise RuntimeError(f"{config.GATEWAY_KEY_ENV} unset — no Labs key (Doppler).")


def run_cycle(
    taskset: FrozenTaskSet,
    *,
    generated_at: str,
    gateway: Any = None,
    judge: Any = None,
) -> CycleResult:
    """Run one cycle. `generated_at` is passed in (no wall-clock in the module)."""
    armset = tuple(s.arm for s in arms.arm_specs())

    # Build the real gateway only when not injected. Injected => test/dry path.
    if gateway is None:
        _assert_live_allowed()
        gateway = GatewayClient()
    judge = judge or GatewayJudge(gateway)
    arms.assert_judge_held_out()  # L3

    specs = {s.arm: s for s in arms.arm_specs()}
    runners = {a: build_live_arm(specs[a], gateway) for a in armset}

    calls: list[ArmCall] = []
    total_cost = 0.0
    halted = False
    for task in taskset.tasks:
        for a in armset:
            out = runners[a].produce(task)
            label = judge.label(task=task, arm=a, output=out.content)
            total_cost += out.cost_usd
            call = ArmCall(
                arm=a,
                task_id=task["id"],
                task_type=task["task_type"],
                success=label.success,
                cost_usd=out.cost_usd,
                tenant=config.LABS_TENANT,
                excluded_from_training=True,
                eval_run_tag=config.EVAL_RUN_TAG,
                judge_score=label.score,
            )
            arms.assert_excluded(call)  # L2 on every call
            calls.append(call)
            if total_cost >= config.COST_CAP_USD:
                halted = True
                break
        if halted:
            break

    arm_metrics = metrics.compute_all(calls, arms=armset)
    type_table = metrics.by_arm_type(calls)
    verdict = metrics.win_check(calls)
    web = build_web_snapshot(
        version=taskset.version,
        generated_at=generated_at,
        arm_metrics=arm_metrics,
        type_table=type_table,
    )
    private = {
        "version": taskset.version,
        "generated_at": generated_at,
        "arm_metrics": {a: m.__dict__ for a, m in arm_metrics.items()},
        "win": verdict.__dict__,
        "n_calls": len(calls),
        "cost_usd": round(total_cost, 6),
    }
    artifacts = write_artifacts(web_snapshot=web, private_results=private)

    if verdict.drift_types:
        log.warning("DRIFT: arm C below the A-measured floor on %s — page founder.",
                    ", ".join(verdict.drift_types))

    return CycleResult(
        status="halted_cost_cap" if halted else "completed",
        n_calls=len(calls),
        cost_usd=round(total_cost, 6),
        arms=armset,
        win=verdict.win,
        drift_types=verdict.drift_types,
        web_snapshot=web,
        artifacts=artifacts,
        note="cost cap hit" if halted else "cycle complete",
    )
