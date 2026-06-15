"""taskset.py — frozen, hashed, version-pinned held-out task set.

The eval runs against a held-out sample frozen at a point in time so a result is
reproducible and tamper-evident. Stage 1 ships the freeze/hash/version-pin
machinery + a small synthetic fixture; Stage 2 (AIN-459) freezes >=200 tasks/type
from a held-out `routing_outcomes` sample on the Labs tenant.

A FrozenTaskSet is identified by (version, hash). `load_frozen` recomputes the
hash and refuses to load if it drifts from the manifest — so a silent edit to the
task file is caught.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labs.eval_harness import config


@dataclass(frozen=True)
class FrozenTaskSet:
    version: str
    hash: str
    n: int
    by_type: dict[str, int]
    tasks: tuple[dict[str, Any], ...]


def canonical_hash(tasks: list[dict[str, Any]]) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace). Stable across
    runs and machines → the CRN integrity anchor for the set."""
    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _by_type(tasks: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t["task_type"]] = counts.get(t["task_type"], 0) + 1
    return dict(sorted(counts.items()))


def freeze(tasks: list[dict[str, Any]], *, version: str) -> FrozenTaskSet:
    """Freeze a task list into a hashed, version-pinned set."""
    return FrozenTaskSet(
        version=version,
        hash=canonical_hash(tasks),
        n=len(tasks),
        by_type=_by_type(tasks),
        tasks=tuple(tasks),
    )


class TaskSetIntegrityError(ValueError):
    """Raised when a loaded task file's hash drifts from its manifest."""


def load_frozen(path: str | Path) -> FrozenTaskSet:
    """Load a `{version, tasks, [hash]}` JSON file and verify its hash.

    If the file carries a `hash`, it must match the recomputed canonical hash or
    we fail closed (the set has been edited since it was frozen).
    """
    data = json.loads(Path(path).read_text())
    version = data["version"]
    tasks = data["tasks"]
    frozen = freeze(tasks, version=version)
    declared = data.get("hash")
    if declared is not None and declared != frozen.hash:
        raise TaskSetIntegrityError(
            f"task-set {version} hash drift: manifest {declared} != recomputed "
            f"{frozen.hash}. The frozen set was edited — fail closed."
        )
    return frozen


def assert_min_per_type(
    frozen: FrozenTaskSet, minimum: int | None = None
) -> list[str]:
    """Return task types that fall short of the per-type minimum.

    Stage 2 treats a non-empty result as a hard fail (need >=200/type); Stage 1
    fixtures are intentionally small, so callers log rather than raise.
    """
    minimum = config.MIN_TASKS_PER_TYPE if minimum is None else minimum
    return sorted(tt for tt, n in frozen.by_type.items() if n < minimum)
