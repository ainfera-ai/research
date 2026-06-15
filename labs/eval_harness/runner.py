"""runner.py — the 4-arm eval runner (interface; INERT in Stage 1).

Orchestrates: freeze task set -> run arms A/B/C/D on the Labs tenant -> judge-label
(held out) -> compute metrics + win check -> emit a weekly snapshot for the
founder sanitization gate.

Stage 1 makes NO live model calls. `run_eval` validates the whole config against
the guardrails, confirms the harness is flag-gated, and returns a dry-run summary.
Passing `live=True` raises — live wiring is Stage 2 (AIN-459).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from labs.eval_harness import arms, config
from labs.eval_harness.judge import StubJudge
from labs.eval_harness.taskset import FrozenTaskSet, assert_min_per_type

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class HarnessResult:
    status: str                       # "disabled" | "dry_run"
    arms: tuple[str, ...]
    judge_model: str
    taskset_version: str | None = None
    taskset_hash: str | None = None
    n_tasks: int | None = None
    undersized_types: tuple[str, ...] = field(default_factory=tuple)
    note: str = ""


def validate_config() -> None:
    """Assert the standing guardrails before anything runs (fail closed)."""
    specs = arms.arm_specs()
    # L3 — judge held out of every arm.
    arms.assert_judge_held_out(specs)
    # Constructing the judge re-checks L3 and pins the held-out identity.
    StubJudge()
    # L2 — exclusion is contract, not toggle.
    if not (config.EXCLUDE_FROM_ROUTING_OUTCOMES and config.EXCLUDE_FROM_TRAINING):
        raise AssertionError("L2: eval calls must be excluded from routing_outcomes + training.")
    # Sanity: exactly the four arms, C is the routed (unpinned) one.
    by_arm = {s.arm: s for s in specs}
    if set(by_arm) != {"A", "B", "C", "D"}:
        raise AssertionError(f"expected arms A/B/C/D, got {sorted(by_arm)}")
    if not by_arm["C"].routed or by_arm["C"].models:
        raise AssertionError("arm C must be the routed (unpinned) arm — L1.")


def run_eval(taskset: FrozenTaskSet | None = None, *, live: bool = False) -> HarnessResult:
    """Run one eval cycle. Stage 1: validate + dry-run only; never live."""
    armset = tuple(s.arm for s in arms.arm_specs())

    if not config.harness_enabled():
        log.info("eval_harness disabled; set %s=true to enable (Stage 2).", config.ENABLED_ENV)
        return HarnessResult(
            status="disabled",
            arms=armset,
            judge_model=config.JUDGE_MODEL,
            note=f"flag {config.ENABLED_ENV} is OFF — Stage 1 ships INERT.",
        )

    # Guardrails first — even when enabled, fail closed if anything is off.
    validate_config()

    if live:
        raise RuntimeError(
            "run_eval(live=True): live model calls land in Stage 2 (AIN-459). "
            "Stage 1 is the interface + metrics contract only."
        )

    undersized: tuple[str, ...] = ()
    version = ts_hash = None
    n_tasks = None
    if taskset is not None:
        version, ts_hash, n_tasks = taskset.version, taskset.hash, taskset.n
        undersized = tuple(assert_min_per_type(taskset))
        if undersized:
            log.warning(
                "task-set %s below %d/type for: %s (Stage-1 fixture; Stage 2 enforces).",
                version, config.MIN_TASKS_PER_TYPE, ", ".join(undersized),
            )

    return HarnessResult(
        status="dry_run",
        arms=armset,
        judge_model=config.JUDGE_MODEL,
        taskset_version=version,
        taskset_hash=ts_hash,
        n_tasks=n_tasks,
        undersized_types=undersized,
        note="config validated; no live calls (Stage 1).",
    )
