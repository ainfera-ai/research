"""taskset.py — frozen tool-use task set loader with hash verification.

A FrozenTaskSet is identified by (version, hash). load_frozen recomputes
the hash and refuses to load if it drifts from the manifest — so a silent
edit to the task file is caught.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RulerTaskSet:
    version: str
    hash: str
    n: int
    tasks: tuple[dict[str, Any], ...]


def canonical_hash(tasks: list[dict[str, Any]]) -> str:
    """SHA-256 over canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class TaskSetIntegrityError(ValueError):
    """Raised when a loaded task file's hash drifts from its manifest."""


def load_frozen(path: str | Path) -> RulerTaskSet:
    """Load a {version, tasks, [hash]} JSON file and verify its hash."""
    data = json.loads(Path(path).read_text())
    version = data["version"]
    tasks = data["tasks"]
    frozen = RulerTaskSet(
        version=version,
        hash=canonical_hash(tasks),
        n=len(tasks),
        tasks=tuple(tasks),
    )
    declared = data.get("hash")
    if declared is not None and declared != frozen.hash:
        raise TaskSetIntegrityError(
            f"task-set {version} hash drift: manifest {declared} != recomputed "
            f"{frozen.hash}. The frozen set was edited — fail closed."
        )
    return frozen
