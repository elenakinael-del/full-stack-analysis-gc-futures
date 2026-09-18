"""
garch.py — GARCH(1,1) conditional volatility.

FIXED:
  1. Crashed with ModuleNotFoundError: no 'arch'. Now checks upfront and
     tells you the exact install command for YOUR interpreter.
  2. No os.makedirs — crashed on savefig if run outside run_all.py.
  3. No convergence check on the fit.
  4. It only drew a picture. It never printed the one number the rest of the
     toolkit needs: today's conditional vol and where it sits historically.
     Now writes garch_state.csv for gameplan.py to consume.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

try:
    from arch import arch_model
    from arch.univariate.base import DataScaleWarning
except ImportError:
    print(f"  MISSING DEPENDENCY: 'arch'\n"
          f"  Install with:  {sys.executable} -m pip install arch", file=sys.stderr)
    sys.exit(1)


def main():
    df = fetch("GC=F", period="max", interval="1d")
    df["returns"] = df["close"].pct_change() * 100
    df = df.dropna()
    if len(df) < 500:
        raise DataError(f"{len(df)} bars, need 500+ for GARCH")

    res = arch_model(df["returns"], vol="Garch", p=1, q=1).fit(disp="off")
    if not res.convergence_flag == 0:
        raise DataError(f"GARCH did not converge (flag {res.convergence_flag}) — "
                        f"do not use this volatility estimate")

    df["volatility"] = res.conditional_volatility
    cur = float(df["volatility"].iloc[-1])
    pct = float((df["volatility"] < cur).mean())
    lr = float(df["volatility"].mean())
    fc = float(np.sqrt(res.forecast(horizon=5, reindex=False).variance.values[-1].mean()))

    a, b = res.params.get("alpha[1]", np.nan), res.params.get("beta[1]", np.nan)
    print(f"  observations    : {len(df):,}  [{df.index[0].date()} -> {df.index[-1].date()}]")
    print(f"  alpha+beta      : {a+b:.4f}  (persistence; >0.99 = near-integrated)")
    print(f"  current vol     : {cur:.2f}% daily  ({cur*np.sqrt(252):.1f}% annualised)")
    print(f"  percentile      : {pct:.1%} of history")
    print(f"  long-run mean   : {lr:.2f}% daily")
    print(f"  5d forecast     : {fc:.2f}% daily  ({'rising' if fc > cur else 'falling'})")
    regime = ("HIGH — reduce size, widen stops" if pct > 0.75 else
              "LOW — normal size, tighter stops" if pct < 0.25 else "NORMAL")
    print(f"  regime          : {regime}")

    pd.DataFrame([dict(asof=df.index[-1].date(), vol_daily=cur,
                       vol_annual=cur*np.sqrt(252), percentile=pct,
                       long_run=lr, forecast_5d=fc, regime=regime,
                       persistence=a+b)]).to_csv(OUT / "garch_state.csv", index=False)

    plt.figure(figsize=(13, 5))
    plt.plot(df.index, df["returns"], label="Returns (%)", alpha=0.35, color="gray", lw=0.5)
    plt.plot(df.index, df["volatility"], label="GARCH conditional volatility", color="red", lw=1)
    plt.axhline(lr, color="k", ls=":", lw=0.8, label=f"long-run {lr:.2f}%")
    plt.title(f"GARCH(1,1) on Gold — current {cur:.2f}%/day ({pct:.0%}ile)")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "garch_vol.png", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  saved -> {OUT}/garch_vol.png, garch_state.csv")


if __name__ == "__main__":
    main()
