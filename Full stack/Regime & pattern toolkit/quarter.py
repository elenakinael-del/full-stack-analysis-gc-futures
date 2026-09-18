"""
quarter.py — intraday seasonality.

FIXED:
  1. It averaged RAW PRICE across a sample where gold ran 4,000 -> 5,300 ->
     4,400. The upward slope from 06:00 to 17:00 was guaranteed by the trend,
     not by time of day. Now averages cumulative return from each session's
     open, then averages across sessions.
  2. It printed "computed intraday volatility" while computing mean price,
     and said "quarter-hour" on 5-minute bars. Now does both, correctly
     labelled, at true quarter-hour granularity.
  3. It localized an already-tz-aware index to New York. yfinance returns
     GC=F intraday tz-aware; forcing a localize is a silent hours-long shift.
     Now only localizes if genuinely naive, and PRINTS which case it hit.
  4. Adds a t-stat so you can see which windows are actually significant
     rather than eyeballing a line.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

FREQ = "15min"


def main():
    import yfinance as yf
    raw = yf.download("GC=F", period="60d", interval="5m",
                      progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise DataError("no intraday data returned")

    tz_before = raw.index.tz
    print(f"  yfinance index tz : {tz_before}")
    if tz_before is None:
        print("  -> index was NAIVE, localising to America/New_York")
    else:
        print("  -> index already tz-aware, converting directly (no localize)")

    close = raw["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    d = pd.DataFrame({"close": close.dropna()})
    d.index = to_athens(d.index)

    d["session"] = d.index.date
    d["tod"] = d.index.floor(FREQ).time
    d["ret"] = d.groupby("session")["close"].pct_change()
    first = d.groupby("session")["close"].transform("first")
    d["cum"] = d["close"] / first - 1.0

    n_sess = d["session"].nunique()
    if n_sess < 20:
        raise DataError(f"only {n_sess} sessions — need 20+ for a profile")

    prof = (d.groupby("tod")
              .agg(mean_cum=("cum", "mean"), se=("cum", "sem"),
                   vol=("ret", lambda s: s.abs().mean()), n=("ret", "count")))
    prof["t"] = prof["mean_cum"] / prof["se"].replace(0, np.nan)
    prof["sig"] = prof["t"].abs() > 2

    print(f"  sessions          : {n_sess}   granularity: {FREQ} (Athens)")
    print(f"  peak vol window   : {prof['vol'].idxmax()}  ({prof['vol'].max()*1e4:.1f} bp/bar)")
    print(f"  quiet vol window  : {prof['vol'].idxmin()}  ({prof['vol'].min()*1e4:.1f} bp/bar)")
    sig = prof[prof["sig"]]
    if len(sig):
        print(f"  significant drift : {len(sig)} windows (|t|>2), "
              f"strongest {sig['t'].abs().idxmax()} t={sig['t'].abs().max():.1f}")
    else:
        print("  significant drift : NONE (|t|<2 everywhere) — no tradable time-of-day edge")

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    x = range(len(prof))
    ax[0].plot(x, prof["mean_cum"] * 100, color="purple", lw=1.4)
    ax[0].fill_between(x, (prof["mean_cum"] - 2 * prof["se"]) * 100,
                       (prof["mean_cum"] + 2 * prof["se"]) * 100,
                       color="purple", alpha=0.15, label="±2 SE")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_ylabel("Mean cumulative return from open (%)")
    ax[0].set_title(f"Gold Intraday Seasonality — Athens time, {n_sess} sessions "
                    f"(return-based, drift-free)")
    ax[0].legend()
    ax[1].bar(x, prof["vol"] * 1e4, color="steelblue")
    ax[1].set_ylabel("Mean |return| (bp)"); ax[1].set_xlabel("Time of day (Athens)")
    step = max(1, len(prof) // 24)
    ax[1].set_xticks(list(x)[::step])
    ax[1].set_xticklabels([str(t)[:5] for t in prof.index[::step]], rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT / "seasonality_2.png", dpi=200)
    plt.close()

    prof.to_csv(OUT / "seasonality_profile.csv")
    print(f"  saved -> {OUT}/seasonality_2.png, seasonality_profile.csv")


if __name__ == "__main__":
    main()
