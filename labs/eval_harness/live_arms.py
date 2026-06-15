"""live_arms.py — real arm runners (Stage 2). Inert until the cycle is enabled.

Each arm turns a task into one ArmOutput (content + total cost + routed flag).
The judge labels success separately (judge.py); cost-per-success is then derived
in metrics.py. Three shapes:

    PinnedArm  (A, B)  — one pinned model passthrough.
    RouterArm  (C)     — model="ainfera-inference" (auto-route; probe agent).
    PanelArm   (D)     — fan out to N pinned members, synthesize with the
                         configured synthesizer; cost is the sum of all calls.

These call the gateway seam only; nothing here imports the SDK directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from labs.eval_harness.arms import ArmSpec
from labs.eval_harness.gateway import GatewayClient, ROUTED_MODEL


@dataclass(frozen=True)
class ArmOutput:
    arm: str
    content: str
    cost_usd: float
    model_used: str
    routed: bool


def _messages(task: dict[str, Any]) -> list[dict[str, str]]:
    return [{"role": "user", "content": task["prompt"]}]


def _synth_prompt(task: dict[str, Any], drafts: list[str]) -> str:
    """Neutral fusion prompt — the real rubric, if any, stays Labs-private."""
    joined = "\n\n".join(f"[Candidate {i + 1}]\n{d}" for i, d in enumerate(drafts))
    return (
        f"Task:\n{task['prompt']}\n\n"
        f"Candidate answers from a panel:\n{joined}\n\n"
        "Synthesize the single best answer to the task using the candidates."
    )


class PinnedArm:
    """A premium/cheap pin (arm A or B)."""

    def __init__(self, spec: ArmSpec, gw: GatewayClient) -> None:
        self.spec, self.gw = spec, gw

    def produce(self, task: dict[str, Any]) -> ArmOutput:
        model = self.spec.models[0]
        r = self.gw.call(model=model, messages=_messages(task), task_type=task.get("task_type"))
        return ArmOutput(self.spec.arm, r.content, r.cost_usd, r.model_used, routed=False)


class RouterArm:
    """Arm C — the outcome-aware router under test (auto-routes)."""

    def __init__(self, spec: ArmSpec, gw: GatewayClient) -> None:
        self.spec, self.gw = spec, gw

    def produce(self, task: dict[str, Any]) -> ArmOutput:
        r = self.gw.call(
            model=ROUTED_MODEL, messages=_messages(task), task_type=task.get("task_type")
        )
        return ArmOutput(self.spec.arm, r.content, r.cost_usd, r.model_used, routed=True)


class PanelArm:
    """Arm D — fan out to panel members, synthesize. Cost = sum of all calls."""

    def __init__(self, spec: ArmSpec, gw: GatewayClient) -> None:
        self.spec, self.gw = spec, gw

    def produce(self, task: dict[str, Any]) -> ArmOutput:
        cost = 0.0
        drafts: list[str] = []
        for member in self.spec.models:
            r = self.gw.call(model=member, messages=_messages(task), task_type=task.get("task_type"))
            cost += r.cost_usd
            drafts.append(r.content)
        synth = self.gw.call(
            model=self.spec.synthesizer or "",
            messages=[{"role": "user", "content": _synth_prompt(task, drafts)}],
            task_type=task.get("task_type"),
        )
        cost += synth.cost_usd
        return ArmOutput(self.spec.arm, synth.content, cost, synth.model_used, routed=False)


def build_live_arm(spec: ArmSpec, gw: GatewayClient):
    """Pick the arm runner shape from the spec."""
    if spec.routed:
        return RouterArm(spec, gw)
    if spec.synthesizer:
        return PanelArm(spec, gw)
    return PinnedArm(spec, gw)
