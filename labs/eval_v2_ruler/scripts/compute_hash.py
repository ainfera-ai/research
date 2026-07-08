#!/usr/bin/env python3
"""Compute the canonical hash for the tool_use_tasks.json fixture."""
import json, hashlib
from pathlib import Path

p = Path(__file__).parent.parent / "fixtures" / "tool_use_tasks.json"
data = json.loads(p.read_text())
tasks = data["tasks"]
canonical = json.dumps(tasks, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
h = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
print(f"hash: {h}")
print(f"n_tasks: {len(tasks)}")

# Write the hash back into the file
data["hash"] = h
p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
print(f"hash written to {p}")
