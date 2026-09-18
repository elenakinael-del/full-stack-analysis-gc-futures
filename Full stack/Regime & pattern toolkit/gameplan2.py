"""
gameplan2.py — signal generator.

FIXED — the first item is the reason this file needed rewriting at all:

  1. FABRICATED OUTPUT. Lines 68-69 of the original were string literals:
         "Best Shape Match: Cluster Type #3 (Range-Break-Trend Blueprint)"
         "Confidence Score: 89.4% similarity to historical template."
     Nothing in the file computed a similarity or a cluster. That 89.4%
     printed on every run regardless of the market. The session-timing block
     was hardcoded text too. Both are now COMPUTED or omitted.

  2. Signal was price vs a 20-bar SMA on 5-minute data — a 100-minute average.
     It flips constantly and has no edge. Replaced with a multi-horizon trend
     check that reports agreement rather than a single crossing.

  3. "Pipeline executed successfully" printed unconditionally.
"""
from _common import *


def main():
    print("=" * 68)
    print("  GOLD SIGNAL GENERATOR")
    print("=" * 68)

    daily = fetch("GC=F", period="5y", interval="1d")
    intra = fetch("GC=F", period="60d", interval="5m")
    dc = daily["close"].dropna()
    ic = intra["close"].dropna()

    # --- [1] volatility regime (computed) ---------------------------------
    r = dc.pct_change().dropna() * 100
    recent, longrun = r.tail(30).std(), r.std()
    ratio = recent / longrun
    if ratio > 1.1:
        vol_status, advice = "ELEVATED", "Cut size ~30%, widen stops."
    elif ratio < 0.9:
        vol_status, advice = "SUPPRESSED", "Normal size; expect expansion."
    else:
        vol_status, advice = "NORMAL", "Standard sizing."
    print(f"\n[1] VOLATILITY REGIME")
    print(f"    30d {recent:.2f}%/day vs 5y {longrun:.2f}%/day  (ratio {ratio:.2f})")
    print(f"    {vol_status} — {advice}")

    # --- [2] structural match: COMPUTED, or stated as unavailable ---------
    print(f"\n[2] STRUCTURAL MATCH")
    tp = OUT / "template_scores.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        cur, best = float(t["ncc"].iloc[-1]), float(t["ncc"].max())
        pct = float((t["ncc"] < cur).mean())
        print(f"    Template NCC (live): {cur:+.3f}   best in sample {best:+.3f}")
        print(f"    Current window is at the {pct:.0%} percentile of all windows")
        print(f"    Verdict: " + ("MATCH" if cur >= 0.70 else "no template match"))
    else:
        print(f"    NOT AVAILABLE — run template.py first.")
        print(f"    (The old version printed '89.4% similarity' here with nothing behind it.)")

    # --- [3] signal from multi-horizon agreement --------------------------
    d = pd.DataFrame({"close": ic})
    d.index = to_athens(d.index)
    px = float(np.asarray(d["close"])[-1])
    horizons = {"1h": 12, "4h": 48, "1d": 288}
    votes, detail = 0, []
    for name, n in horizons.items():
        if len(d) <= n: continue
        ma = float(d["close"].rolling(n).mean().iloc[-1])
        v = 1 if px > ma else -1
        votes += v
        detail.append(f"{name} {'>' if v>0 else '<'} MA({ma:,.1f})")

    print(f"\n[3] TREND AGREEMENT")
    print(f"    Price ${px:,.2f}   " + " | ".join(detail))
    if votes == len(detail) and votes > 0:
        sig, why = "LONG", "price above every horizon — aligned uptrend"
    elif votes == -len(detail) and votes < 0:
        sig, why = "SHORT", "price below every horizon — aligned downtrend"
    else:
        sig, why = "FLAT", f"horizons disagree ({votes:+d}) — no edge"
    print(f"    SIGNAL: {sig}   ({why})")

    # --- [4] session windows: measured from data, not typed in ------------
    print(f"\n[4] INTRADAY WINDOWS (measured, Athens time)")
    sp = OUT / "seasonality_profile.csv"
    if sp.exists():
        s = pd.read_csv(sp, index_col=0)
        top = s.nlargest(3, "vol")
        print(f"    Highest-volatility windows:")
        for tod, row in top.iterrows():
            print(f"      {str(tod)[:5]}  {row['vol']*1e4:.1f} bp/bar")
        if "sig" in s and s["sig"].any():
            print(f"    Windows with significant drift (|t|>2): {int(s['sig'].sum())}")
        else:
            print(f"    No time-of-day window shows significant drift (|t|<2 everywhere).")
    else:
        print(f"    NOT AVAILABLE — run quarter.py first.")
        print(f"    (The old version printed fixed ET windows that were never measured.)")

    print("\n" + "=" * 68)
    print(f"  SUMMARY: {sig} | vol {vol_status} | {advice}")
    print("=" * 68)


if __name__ == "__main__":
    main()
