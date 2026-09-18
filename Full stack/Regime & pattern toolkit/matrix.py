"""
matrix.py — matrix profile motif discovery.

FIXED:
  1. Crashed with ModuleNotFoundError: no 'stumpy'.
  2. end="2026-08-15" was HARDCODED. It would silently stop updating.
  3. max_distance=float('inf') accepted ANY window as a motif match — that is
     why the "match" looked arbitrary. Now uses stumpy's default threshold and
     reports the actual distance so you can judge it.
  4. Ran on raw price 250 -> 5,300. Now runs on log price so a 30-day shape
     in 2008 is comparable to one in 2026.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

try:
    import stumpy
except ImportError:
    print(f"  MISSING DEPENDENCY: 'stumpy'\n"
          f"  Install with:  {sys.executable} -m pip install stumpy", file=sys.stderr)
    sys.exit(1)

WINDOW = 30


def main():
    df = fetch("GC=F", start="2000-01-01")        # no hardcoded end date
    close = df["close"].dropna()
    dates = close.index
    series = np.log(close.values).astype(float)   # log space
    print(f"  bars            : {len(series):,}  [{dates[0].date()} -> {dates[-1].date()}]")

    mp = stumpy.stump(series, m=WINDOW)
    prof = mp[:, 0].astype(float)

    d, idx = stumpy.motifs(pd.Series(series), prof, max_motifs=3, max_matches=5)

    print(f"  window          : {WINDOW} days on log price")
    print(f"  profile dist    : min {np.nanmin(prof):.3f}  median {np.nanmedian(prof):.3f}  "
          f"max {np.nanmax(prof):.3f}")

    for i, (dist_row, idx_row) in enumerate(zip(d, idx)):
        valid = [(dd, ii) for dd, ii in zip(dist_row, idx_row) if ii != -1 and np.isfinite(dd)]
        if not valid: continue
        print(f"\n  motif {i+1}:")
        for dd, ii in valid:
            print(f"    {dates[ii].date()}  distance {dd:.3f}")

    # is TODAY's window like anything in history?
    cur = len(prof) - 1
    nn = int(mp[cur, 1])
    print(f"\n  current window ({dates[cur].date()} -> {dates[-1].date()}):")
    print(f"    nearest historical analogue: {dates[nn].date()}  distance {prof[cur]:.3f}")
    rank = float((prof < prof[cur]).mean())
    print(f"    that distance is at the {rank:.0%} percentile of all windows")
    if rank > 0.5:
        print("    -> current shape is UNUSUAL; the analogue is weak, do not lean on it")
    else:
        print("    -> current shape has a genuinely close historical match")

    fig, ax = plt.subplots(2, sharex=True, figsize=(13, 7))
    ax[0].plot(dates, close.values, color="teal", lw=0.8, label="Gold GC=F Close")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Price (log)")
    ax[0].set_title("Matrix Profile — Gold Repeating Pattern Discovery")
    if len(idx) and len([x for x in idx[0] if x != -1]) >= 2:
        v = [x for x in idx[0] if x != -1][:2]
        ax[0].axvline(dates[v[0]], color="red", ls="--", label="Motif 1")
        ax[0].axvline(dates[v[1]], color="green", ls="--", label="Match")
    ax[0].axvline(dates[cur], color="orange", ls="-", lw=1.2, label="Now")
    ax[0].legend(fontsize=8)
    ax[1].plot(dates[:len(prof)], prof, color="orange", lw=0.6)
    ax[1].set_ylabel("Matrix Profile Distance")
    plt.tight_layout()
    plt.savefig(OUT / "matrix_profile_2.jpg", dpi=200)
    plt.close()
    print(f"\n  saved -> {OUT}/matrix_profile_2.jpg")


if __name__ == "__main__":
    main()
