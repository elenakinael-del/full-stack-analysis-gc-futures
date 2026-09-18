"""
template.py — template matching by cross-correlation.

FIXED:
  1. The normalisation was wrong. It divided by the GLOBAL std of the whole
     price series, so scores were neither bounded to [-1,1] nor comparable
     between windows — which is why the output looked like noise with one
     spike. Now z-normalises EACH window (true NCC).
  2. It matched on raw price levels. Now matches on the shape of returns.
  3. best_match_idx was printed as a bare integer. Now prints the DATE and
     the correlation value, and flags whether it clears a significance bar.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

# V-shaped reversal: down, trough, recovery
TEMPLATE = np.array([1.0, 0.5, -0.5, -1.0, -0.5, 0.5, 1.0])
MIN_CORR = 0.70


def ncc(series, template):
    """True normalised cross-correlation: z-normalise each window."""
    t = znorm(template); m = len(t)
    out = np.full(len(series) - m + 1, np.nan)
    for i in range(len(out)):
        w = znorm(series[i:i + m])
        out[i] = float(np.dot(w, t) / m)
    return out


def main():
    df = fetch("GC=F", period="1y", interval="1d").dropna()
    px = df["close"].values
    ret = np.diff(np.log(px))          # shape of RETURNS, not levels
    dates = df.index[1:]

    corr = ncc(ret, TEMPLATE)
    m = len(TEMPLATE)
    best = int(np.nanargmax(corr))
    worst = int(np.nanargmin(corr))

    print(f"  window          : {m} days, matching on log-return shape")
    print(f"  best match      : {dates[best].date()} -> {dates[best+m-1].date()}  "
          f"NCC {corr[best]:+.3f}")
    print(f"  worst (inverse) : {dates[worst].date()}  NCC {corr[worst]:+.3f}")
    print(f"  current window  : NCC {corr[-1]:+.3f}  "
          f"({'MATCH' if corr[-1] >= MIN_CORR else 'no match'})")

    hits = np.where(corr >= MIN_CORR)[0]
    print(f"  windows >= {MIN_CORR}: {len(hits)} of {len(corr)} ({len(hits)/len(corr):.1%})")
    if len(hits) == 0:
        print("  -> template does not occur in this sample; do not read the peak as a signal")

    plt.figure(figsize=(13, 5))
    plt.plot(dates[:len(corr)], corr, color="purple", lw=1, label="NCC")
    plt.axhline(MIN_CORR, color="green", ls=":", label=f"match bar {MIN_CORR}")
    plt.axhline(-MIN_CORR, color="red", ls=":")
    plt.axvline(dates[best], color="red", ls="--", label=f"best {corr[best]:+.2f}")
    plt.ylim(-1, 1)
    plt.title("Template Matching — normalised cross-correlation on gold return shape")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "template_match.jpg", dpi=200, bbox_inches="tight")
    plt.close()

    pd.DataFrame({"date": dates[:len(corr)], "ncc": corr}).to_csv(
        OUT / "template_scores.csv", index=False)
    print(f"  saved -> {OUT}/template_match.jpg, template_scores.csv")


if __name__ == "__main__":
    main()
