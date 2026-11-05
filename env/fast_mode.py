"""Runtime switch for the reduced 'first results in ~1 hour' plan (env/fast_mode.json).

Queued Slurm jobs keep a copy of their batch script from submission time, but the Python scripts they
call are read when the job runs. Each script calls these helpers and exits 0 for disabled steps, so
`set -e` job scripts and `afterok` dependencies still succeed. Delete fast_mode.json to undo.
"""
import json
import re
import sys
from pathlib import Path

CFG_PATH = Path(__file__).resolve().parent / "fast_mode.json"


def cfg():
    return json.load(open(CFG_PATH)) if CFG_PATH.exists() else {}


def skip(reason):
    print(f"[fast_mode] skipping: {reason} (see {CFG_PATH})", flush=True)
    sys.exit(0)


def check_steps(steps_or_path):
    """Exit 0 if this step count (int, or a path containing '<n>steps') is disabled."""
    c = cfg()
    if "steps" not in c:
        return
    m = re.search(r"(\d+)steps", str(steps_or_path)) if not isinstance(steps_or_path, int) else None
    n = steps_or_path if isinstance(steps_or_path, int) else (int(m.group(1)) if m else None)
    if n is not None and n not in c["steps"]:
        skip(f"{n}-step set disabled")
