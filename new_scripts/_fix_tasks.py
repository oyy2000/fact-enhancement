#!/usr/bin/env python3
"""Unskip all tasks, reset failed to pending, and print status."""
import json
from pathlib import Path

TASKS = Path("/common/users/sl2148/Public/yang_ouyang/projects/fact-enhancement/new_exps/tasks.json")

with open(TASKS) as f:
    data = json.load(f)

changes = 0
for t in data["tasks"]:
    if t.get("skip"):
        t["skip"] = False
        if t["status"] != "done":
            t["status"] = "pending"
            changes += 1
    if t["status"] == "failed":
        t["status"] = "pending"
        t["attempts"] = 0
        changes += 1

with open(TASKS, "w") as f:
    json.dump(data, f, indent=2)

pending = [t for t in data["tasks"] if t["status"] == "pending"]
done = [t for t in data["tasks"] if t["status"] == "done"]

print(f"Changes: {changes}")
print(f"Done: {len(done)}, Pending: {len(pending)}")
print("\nPending tasks:")
for t in pending:
    short = t["model"].split("/")[-1].replace("-Instruct", "")
    print(f"  {short:25s} x {t['dataset']:10s} TP={t['tp']}")
