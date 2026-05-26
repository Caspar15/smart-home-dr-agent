"""End-to-end runner for the HOURLY variant.

Sets REPRO_RESAMPLE=1h and chains data_prep -> models_classical -> models_lstm
-> evaluate_retrain. Outputs are saved with the _1h suffix.
"""
from __future__ import annotations
import os
import sys
import subprocess

env = dict(os.environ)
env["REPRO_RESAMPLE"] = "1h"
env["PYTHONIOENCODING"] = "utf-8"

here = os.path.dirname(os.path.abspath(__file__))
py = sys.executable

steps = [
    ("data_prep", ["-X", "utf8", "data_prep.py"]),
    ("classical", ["-X", "utf8", "models_classical.py", "--fast"]),
    ("lstm",      ["-X", "utf8", "models_lstm.py"]),
    ("eval",      ["-X", "utf8", "evaluate_retrain.py", "--fast"]),
]

for name, cmd in steps:
    print(f"\n=== running {name} ===")
    full = [py, *cmd]
    r = subprocess.run(full, cwd=here, env=env)
    if r.returncode != 0:
        print(f"!! {name} failed (exit {r.returncode})")
        sys.exit(r.returncode)

print("\n=== hourly pipeline complete ===")
print("Outputs saved to results/*_1h.csv and figures/*")
