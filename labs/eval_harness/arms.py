"""arms.py — the 4 eval arms (A/B/C/D) as interfaces + Stage-1 stubs.

The proof eval compares four arms on cost-per-successful-task:

    A  premium pin     a single premium model, pinned (eval-only)
    B  cheap pin       a single cheap model, pinned (eval-only)
    C  ainfera         the outcome-aware router (the thing under test)
    D  fusion panel    a multi-model panel, synthesized by FUSION_SYNTHESIZER

Stage 1 ships INTERFACES + STUBS only — no arm makes a live model call. Every
`run()` raises NotImplementedError; Stage 2 (AIN-459) wires them on the Labs
tenant.

L1 (locks-2026-06-15-proof-pipeline): arm pins live ONLY here, never in the
production router. Concrete A/B/D model identifiers are env-injected on the Labs
tenant and are NOT committed to this public repo — the defaults below are
placeholders. The concrete roster lives in the private vault eval spec
(decisions/2026-06-15-eval-spec-3arm-cost-outcome.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from labs.eval_harness import config


# --- arm specification -----------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """Static description of one arm. `models` are resolved (env-injected) ids.

    For pinned arms (A/B) `models` has one entry. For the fusion panel (D) it has
    the panel members and `synthesizer` is set. For the router (C) `models` is
    empty — C does not pin; it routes, and `routed` is True.
    """

    arm: str            # "A" | "B" | "C" | "D"
    role: str           # premium_pin | cheap_pin | ainfera_router | fusion_panel
    models: tuple[str, ...]
    routed: bool = False
    synthesizer: str | None = None

    def all_model_refs(self) -> tuple[str, ...]:
        """Every concrete model this arm touches (members + synthesizer)."""
        refs = list(self.models)
        if self.synthesizer:
            refs.append(self.synthesizer)
        return tuple(refs)


@dataclass(frozen=True)
class ArmCall:
    """One arm's outcome on one task. Stage 1 defines the SHAPE; Stage 2 fills it
    from live calls. The tenant/exclusion fields make L2 (Labs-only, fenced out
    of routing_outcomes + training) assertable on every record."""

    arm: str
    task_id: str
    task_type: str
    success: bool
    cost_usd: float
    tenant: str
    excluded_from_training: bool
    eval_run_tag: str
    judge_score: float | None = None


@runtime_checkable
class Arm(Protocol):
    spec: ArmSpec

    def run(self, task: dict[str, Any]) -> ArmCall:
        """Execute the arm on one task. Stage 1: raises NotImplementedError."""
        ...


class StubArm:
    """Inert arm — Stage 1. Holds its spec; refuses to make a live call."""

    def __init__(self, spec: ArmSpec) -> None:
        self.spec = spec

    def run(self, task: dict[str, Any]) -> ArmCall:  # noqa: ARG002 - interface
        raise NotImplementedError(
            f"arm {self.spec.arm} ({self.spec.role}): live wiring lands in Stage 2 "
            f"(AIN-459). Stage 1 is the interface + metrics contract only."
        )


# --- arm registry ----------------------------------------------------------


def _pin(env_key: str, placeholder: str) -> str:
    """Resolve a pinned model id from env (Labs Doppler) — placeholder default so
    no concrete competitor pin is committed to the public repo (L1/L4)."""
    return os.environ.get(env_key, placeholder)


def arm_specs() -> tuple[ArmSpec, ...]:
    """The four arm specs, pins resolved from env (placeholders by default)."""
    panel = tuple(
        m.strip()
        for m in os.environ.get(
            "LABS_EVAL_ARM_D_PANEL",
            "<arm-d-panel-1>,<arm-d-panel-2>,<arm-d-panel-3>",
        ).split(",")
        if m.strip()
    )
    return (
        ArmSpec("A", "premium_pin", (_pin("LABS_EVAL_ARM_A", "<arm-a-premium>"),)),
        ArmSpec("B", "cheap_pin", (_pin("LABS_EVAL_ARM_B", "<arm-b-cheap>"),)),
        ArmSpec("C", "ainfera_router", (), routed=True),
        ArmSpec("D", "fusion_panel", panel, synthesizer=config.FUSION_SYNTHESIZER),
    )


def build_arms(specs: tuple[ArmSpec, ...] | None = None) -> dict[str, Arm]:
    """Instantiate the (stub) arms keyed by arm letter."""
    specs = specs or arm_specs()
    return {s.arm: StubArm(s) for s in specs}


# --- guardrail assertions --------------------------------------------------


class JudgeLeakError(AssertionError):
    """Raised when the held-out judge model appears inside any arm (L3)."""


def assert_judge_held_out(
    specs: tuple[ArmSpec, ...] | None = None, judge_model: str | None = None
) -> None:
    """L3: the judge must never appear as an arm model (member or synthesizer)."""
    specs = specs or arm_specs()
    judge_model = judge_model or config.JUDGE_MODEL
    for s in specs:
        if judge_model in s.all_model_refs():
            raise JudgeLeakError(
                f"judge {judge_model!r} is held out of all arms, but appears in arm "
                f"{s.arm} ({s.role}). L3 violation — fail closed."
            )


def assert_excluded(call: ArmCall) -> None:
    """L2: every eval call must be on the Labs tenant and fenced from training."""
    if call.tenant != config.LABS_TENANT:
        raise AssertionError(
            f"eval call on tenant {call.tenant!r}, expected {config.LABS_TENANT!r} "
            f"(L2 — eval runs on Labs only)."
        )
    if not call.excluded_from_training:
        raise AssertionError("eval call not marked excluded_from_training (L2).")
    if call.eval_run_tag != config.EVAL_RUN_TAG:
        raise AssertionError(
            f"eval call missing {config.EVAL_RUN_TAG!r} tag (L2 defence-in-depth)."
        )
