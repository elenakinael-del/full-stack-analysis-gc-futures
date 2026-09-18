"""
run_all.py — toolkit runner.

FIXED:
  1. THE CAUSE OF YOUR TWO CRASHES. You ran with /usr/bin/python3 (system
     Python), but 'arch' and 'stumpy' live in your Anaconda environment.
     sys.executable then pointed every subprocess at the wrong interpreter.
     This now reports which Python it is using and checks dependencies BEFORE
     running anything.
  2. Scripts were resolved relative to the CWD, so it only worked if you
     happened to be in the right folder. Now resolves relative to this file.
  3. gameplan.py / gameplan2.py were never in the list.
  4. A failure printed "Error executing X" and carried on, then the run
     finished with "All modules executed!" — which read like success.
     Now prints a pass/fail table and exits non-zero if anything failed.
"""
import os, sys, subprocess, importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
(HERE / "outputs").mkdir(exist_ok=True)

# order matters: producers before consumers
SCRIPTS = ["dtw.py", "garch.py", "HSMM.py", "matrix.py", "quarter.py",
           "template.py", "gameplan.py", "gameplan2.py"]

DEPS = {"yfinance": "yfinance", "pandas": "pandas", "numpy": "numpy",
        "matplotlib": "matplotlib", "scipy": "scipy",
        "arch": "arch", "stumpy": "stumpy", "hmmlearn": "hmmlearn"}


def check_deps():
    print(f"  interpreter : {sys.executable}")
    missing = [pip for mod, pip in DEPS.items()
               if importlib.util.find_spec(mod) is None]
    if missing:
        print(f"\n  MISSING: {', '.join(missing)}")
        print(f"  Install into THIS interpreter:\n")
        print(f"      {sys.executable} -m pip install {' '.join(missing)}\n")
        print(f"  If you use Anaconda, run this script with Anaconda's python:")
        print(f"      python run_all.py        (not /usr/bin/python3)")
        return False
    print(f"  dependencies: all present")
    return True


def run(script):
    p = HERE / script
    if not p.exists():
        return "SKIP", "file not found"
    print(f"\n{'='*16} {script} {'='*16}")
    r = subprocess.run([sys.executable, str(p)], capture_output=False)
    return ("OK", "") if r.returncode == 0 else ("FAIL", f"exit {r.returncode}")


if __name__ == "__main__":
    print("=" * 60)
    print("  GC PATTERN DETECTION TOOLKIT")
    print("=" * 60)
    ok = check_deps()
    if not ok:
        print("  Continuing anyway — scripts needing missing packages will fail.\n")

    results = {s: run(s) for s in SCRIPTS}

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for s, (st, note) in results.items():
        mark = {"OK": "  ok  ", "FAIL": " FAIL ", "SKIP": " skip "}[st]
        print(f"  [{mark}] {s:16s} {note}")

    n_fail = sum(1 for st, _ in results.values() if st == "FAIL")
    print("=" * 60)
    if n_fail:
        print(f"  {n_fail} script(s) FAILED. Outputs in 'outputs/' are incomplete —")
        print(f"  do not trade off a partial run.")
        sys.exit(1)
    print(f"  All {len(SCRIPTS)} scripts completed. See 'outputs/'.")
