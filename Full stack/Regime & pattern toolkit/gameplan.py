"""
gameplan.py — synthesis of the toolkit outputs into a bias.

FIXED (the second one is the serious one):
  1. GaussianHMM(covariance_type="full") on ~125 daily returns from 6mo, with
     no convergence check. Same failure as the old HSMM.py.
  2. The bias logic keyed off `current_state != 2`. HMM STATE INDICES ARE
     ARBITRARY — they change every refit. State 2 could be calm-bull today and
     crisis-bear tomorrow. The gameplan was keyed to a meaningless number.
     Now reads HSMM.py's LABELLED output instead of refitting.
  3. `state_mean` was computed and never used.
  4. It printed "Matrix Profile & Template outputs confirm structural
     repetition bounds" unconditionally — a claim about files it never opened.
     Now reads them and reports what they actually say, or says they are missing.
"""
from _common import *


def read_csv(name):
    p = OUT / name
    return pd.read_csv(p) if p.exists() else None


def main():
    print("=" * 62)
    print("  QUANTITATIVE GOLD GAMEPLAN")
    print("=" * 62)

    d = fetch("GC=F", period="6mo", interval="1d")
    px, prev = last(d, "close"), float(np.asarray(d["close"].dropna())[-2])
    chg = (px - prev) / prev * 100
    hi20, lo20 = d["high"].tail(20).max(), d["low"].tail(20).min()
    pos = (px - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5

    print(f"\n[MARKET STATE]")
    print(f"  Price (GC=F)      : ${px:,.2f}   ({chg:+.2f}% on the day)")
    print(f"  20d range         : ${lo20:,.0f} - ${hi20:,.0f}   position {pos:.0%}")

    # --- regime: read HSMM's labelled output, do NOT refit -----------------
    reg = read_csv("gold_regime_summary.csv")
    states = read_csv("gold_regimes.csv")
    label = None
    if states is not None and len(states):
        label = str(states["label"].iloc[-1])
        run = 1
        s = states["state"].values
        for x in s[-2::-1]:
            if x == s[-1]: run += 1
            else: break
        typ = reg.loc[reg.state == s[-1], "obs_dwell"].iloc[0] if reg is not None else np.nan
        print(f"  Regime (HSMM)     : {label}   day {run} of a typical {typ:.0f}-day run")
    else:
        print(f"  Regime (HSMM)     : NOT AVAILABLE — run HSMM.py first")

    # --- volatility: read GARCH output -------------------------------------
    g = read_csv("garch_state.csv")
    if g is not None:
        v = g.iloc[0]
        print(f"  Volatility (GARCH): {v.vol_annual:.1f}% annualised, "
              f"{v.percentile:.0%}ile — {v.regime}")
    else:
        print(f"  Volatility (GARCH): NOT AVAILABLE — run garch.py first")

    # --- structure: actually read the files, don't assert --------------------
    t = read_csv("template_scores.csv")
    print(f"\n[STRUCTURAL EVIDENCE]")
    if t is not None:
        cur_ncc = float(t["ncc"].iloc[-1]); mx = float(t["ncc"].max())
        print(f"  Template NCC now  : {cur_ncc:+.3f}  (best in sample {mx:+.3f})")
        print(f"  Verdict           : " +
              ("template present" if cur_ncc >= 0.70 else "NO template match — ignore"))
    else:
        print(f"  Template          : NOT AVAILABLE — run template.py first")

    dm = OUT / "dtw_matrix.csv"
    if dm.exists():
        M = pd.read_csv(dm, index_col=0)
        md = M.values.sum(axis=1) / (len(M) - 1)
        print(f"  Session shape     : latest mean DTW {md[-1]:.2f} vs median {np.median(md):.2f}"
              f"  ({'unusual' if md[-1] > np.median(md) else 'typical'})")
    else:
        print(f"  Session shape     : NOT AVAILABLE — run dtw.py first")

    # --- bias: built only from evidence that EXISTS -------------------------
    score, reasons = 0, []
    if chg > 0: score += 1; reasons.append(f"day {chg:+.2f}%")
    elif chg < 0: score -= 1; reasons.append(f"day {chg:+.2f}%")
    if label:
        if "bull" in label: score += 1; reasons.append(f"regime {label}")
        elif "bear" in label: score -= 1; reasons.append(f"regime {label}")
    if pos > 0.7: score += 1; reasons.append("upper 20d range")
    elif pos < 0.3: score -= 1; reasons.append("lower 20d range")

    bias = ("BULLISH" if score >= 2 else "BEARISH" if score <= -2 else "RANGE / NO EDGE")
    print(f"\n[BIAS]  {bias}   (score {score:+d} from: {', '.join(reasons) or 'no inputs'})")
    if abs(score) < 2:
        print("  Not enough agreement for a directional bias. Trade the edges or stand aside.")
    if g is not None and float(g.iloc[0].percentile) > 0.75:
        print("  ! Vol in the top quartile — cut size regardless of direction.")
    print("=" * 62)


if __name__ == "__main__":
    main()
