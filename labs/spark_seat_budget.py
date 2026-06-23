"""AIN-548 · Spark Labs seat budgeting + cost report (128 GB DGX Spark, offline).

PURE planner: pack Labs seat requests into the DGX Spark 128 GB envelope — the substrate for
the offline Spark batches (Tier-0 competitor loop AIN-376, distill AIN-561, the nightly
refit) — and emit a cost report so the Labs tenant can decide what fits a window. The actual
scheduling (Slurm / cgroup pinning on the box) is infra; this is the budget math + report.
No I/O, no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SPARK_TOTAL_GB = 128.0
# Owned DGX ⇒ marginal $0; set a rate for amortised accounting in the cost report.
DEFAULT_GPU_COST_PER_HOUR = 0.0


@dataclass(frozen=True)
class SeatRequest:
    name: str
    mem_gb: float
    gpu_hours: float


@dataclass(frozen=True)
class SeatPlan:
    admitted: list[str]
    rejected: list[str]  # did not fit the remaining envelope
    used_gb: float
    free_gb: float
    utilisation: float  # used_gb / total_gb
    total_gpu_hours: float
    total_cost_usd: float


def plan_seats(
    requests: Sequence[SeatRequest],
    *,
    total_gb: float = SPARK_TOTAL_GB,
    gpu_cost_per_hour: float = DEFAULT_GPU_COST_PER_HOUR,
) -> SeatPlan:
    """Greedily admit seats (in priority order) while they fit the 128 GB envelope; reject the
    rest. Returns admitted/rejected + memory utilisation + the GPU-hour cost of the admitted
    set. A single seat larger than the whole envelope is simply rejected (never admitted)."""
    remaining = total_gb
    admitted: list[str] = []
    rejected: list[str] = []
    gpu_hours = 0.0
    for r in requests:
        if 0 <= r.mem_gb <= remaining:
            admitted.append(r.name)
            remaining -= r.mem_gb
            gpu_hours += r.gpu_hours
        else:
            rejected.append(r.name)
    used = total_gb - remaining
    return SeatPlan(
        admitted=admitted,
        rejected=rejected,
        used_gb=round(used, 3),
        free_gb=round(remaining, 3),
        utilisation=round(used / total_gb, 6) if total_gb > 0 else 0.0,
        total_gpu_hours=round(gpu_hours, 3),
        total_cost_usd=round(gpu_hours * gpu_cost_per_hour, 4),
    )


__all__ = ["DEFAULT_GPU_COST_PER_HOUR", "SPARK_TOTAL_GB", "SeatPlan", "SeatRequest", "plan_seats"]
