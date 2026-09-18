#!/usr/bin/env python3
"""
validate.py — run this FIRST, every Sunday, before any other notebook.

Exits non-zero if anything that burned you last week is present again.
Wire it into the top of each notebook:

    import subprocess, sys
    if subprocess.run([sys.executable, "validate.py"]).returncode != 0:
        raise SystemExit("Preflight failed — do not trade off this run.")

Or run standalone:   python3 validate.py
"""

from __future__ import annotations

import ast
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

GREEN, RED, YELLOW, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[0m"

FAILURES: list[str] = []
WARNINGS: list[str] = []


def fail(msg: str):
    FAILURES.append(msg)
    print(f"{RED}  [FAIL]{RESET} {msg}")


def warn(msg: str):
    WARNINGS.append(msg)
    print(f"{YELLOW}  [WARN]{RESET} {msg}")


def ok(msg: str):
    print(f"{GREEN}  [ok]  {RESET} {msg}")


# ---------------------------------------------------------------------------
# A. Static scan — catch the bugs before they run
# ---------------------------------------------------------------------------

BANNED_PATTERNS = [
    (r"GC_FUTURES_PRICE\s*=\s*[\d_.]+",
     "hardcoded GC_FUTURES_PRICE — use goldlib.get_context()"),
    (r"(?:GLD|gld|spot)\s*\*\s*10(?!\d|\s*\*\*|\.\d)\b",
     "GLD * 10 — the ratio is ~10.91; use ctx.gld_to_gold()"),
    (r"MULTIPLIER\s*=\s*10\.0*\b",
     "MULTIPLIER = 10.0 — use the live ratio"),
    (r"SUNDAY_BASE\s*=\s*[\d_.]+",
     "hardcoded SUNDAY_BASE — recompute from data"),
    (r"warnings\.filterwarnings\(\s*['\"]ignore['\"]\s*\)",
     "blanket warnings suppression — this is how you missed the "
     "convergence failures; scope it or drop it"),
]


def scan_sources(root: Path):
    print("\nA. STATIC SCAN")
    files = sorted(list(root.glob("*.py")) + list(root.glob("*.ipynb")))
    if not files:
        warn(f"no .py/.ipynb files found under {root}")
        return
    hits = 0
    for f in files:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for pat, msg in BANNED_PATTERNS:
            for m in re.finditer(pat, text):
                line = text[:m.start()].count("\n") + 1
                fail(f"{f.name}:{line} — {msg}")
                hits += 1
    if hits == 0:
        ok(f"{len(files)} file(s) clean of banned patterns")


# ---------------------------------------------------------------------------
# B. Live data
# ---------------------------------------------------------------------------

def check_live():
    print("\nB. LIVE DATA")
    try:
        from goldlib.prices import get_context
    except ImportError:
        fail("goldlib not importable — is it on sys.path?")
        return None
    try:
        ctx = get_context()
    except Exception as e:
        fail(f"price fetch failed: {type(e).__name__}: {e}")
        return None

    ok(f"GC={ctx.gc:,.2f}  GLD={ctx.gld:,.2f}  ratio={ctx.ratio:.4f}x  ({ctx.series})")

    if abs(ctx.ratio - 10.0) < 0.05:
        warn("ratio is suspiciously close to 10.0 — verify it is real, not a stub")

    age_h = (datetime.now(timezone.utc)
             - datetime.fromisoformat(ctx.fetched_at)).total_seconds() / 3600
    if age_h > 1:
        warn(f"context fetched {age_h:.1f}h ago")
    return ctx


# ---------------------------------------------------------------------------
# C. Cross-notebook consistency
# ---------------------------------------------------------------------------

def check_consistency(values: dict[str, dict[str, float]], tol=0.005):
    """
    values: {"metric": {"notebook": value}}
    Every notebook must agree on shared metrics, or you get six flip levels.
    """
    print("\nC. CROSS-NOTEBOOK CONSISTENCY")
    if not values:
        warn("no cross-notebook values registered yet")
        return
    for metric, by_nb in values.items():
        vals = [v for v in by_nb.values() if v is not None]
        if len(vals) < 2:
            continue
        spread = (max(vals) - min(vals)) / abs(max(vals) or 1)
        if spread > tol:
            detail = ", ".join(f"{k}={v:,.2f}" for k, v in by_nb.items())
            fail(f"{metric}: {len(vals)} notebooks disagree by {spread:.1%} — {detail}")
        else:
            ok(f"{metric}: consistent across {len(vals)} notebooks")


# ---------------------------------------------------------------------------
# D. Manual field expiry
# ---------------------------------------------------------------------------

def check_manual_fields(manual: dict, shelf_life_days: int = 7):
    print("\nD. MANUAL FIELDS")
    if not manual:
        warn("no manual fields registered")
        return
    today = datetime.now(timezone.utc).date()
    for name, entry in manual.items():
        as_of = entry.get("as_of") if isinstance(entry, dict) else None
        if not as_of:
            fail(f"'{name}' has no as_of date — cannot tell if it is stale")
            continue
        age = (today - datetime.fromisoformat(as_of).date()).days
        if age > shelf_life_days:
            fail(f"'{name}' written {as_of} ({age}d ago) — EXPIRED, rewrite or delete")
        else:
            ok(f"'{name}' fresh ({age}d)")


# ---------------------------------------------------------------------------
# E. Verdict coherence
# ---------------------------------------------------------------------------

def check_verdicts(verdicts: dict[str, float]):
    """
    Pass every headline score your stack produces. If they disagree in SIGN,
    you do not have a bias — you have a bug. Last week: header -5,
    scorecard +2, manual bias -4.
    """
    print("\nE. VERDICT COHERENCE")
    if len(verdicts) < 2:
        warn("fewer than two verdicts registered")
        return
    signs = {k: (1 if v > 0 else -1 if v < 0 else 0) for k, v in verdicts.items()}
    nonzero = {s for s in signs.values() if s != 0}
    detail = ", ".join(f"{k}={v:+g}" for k, v in verdicts.items())
    if len(nonzero) > 1:
        fail(f"verdicts disagree in sign — {detail}. Resolve before trading.")
    else:
        ok(f"verdicts coherent — {detail}")


# ---------------------------------------------------------------------------

def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print("=" * 68)
    print(f"  PREFLIGHT  |  {datetime.now().strftime('%A %d %B %Y %H:%M')}")
    print(f"  scanning: {root}")
    print("=" * 68)

    scan_sources(root)
    check_live()

    # Register these from your notebooks as they run:
    check_consistency({})
    check_manual_fields({})
    check_verdicts({})

    print("\n" + "=" * 68)
    if FAILURES:
        print(f"{RED}  PREFLIGHT FAILED — {len(FAILURES)} error(s), "
              f"{len(WARNINGS)} warning(s){RESET}")
        print(f"{RED}  DO NOT TRADE OFF THIS RUN.{RESET}")
        print("=" * 68)
        return 1
    print(f"{GREEN}  PREFLIGHT PASSED{RESET}  ({len(WARNINGS)} warning(s))")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
