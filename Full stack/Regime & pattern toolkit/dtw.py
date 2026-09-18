"""
dtw.py — session-shape distance matrix.

FIXED:
  1. It was not DTW. Line 25 was np.linalg.norm(s1-s2) — plain Euclidean.
     Now uses a real DTW with a Sakoe-Chiba band.
  2. It compared RAW PRICE LEVELS. Two identically-shaped sessions 100 points
     apart scored as maximally distant, so the matrix clustered by price era,
     not by shape. Each session is now z-normalised first.
  3. s[:20] kept only the first 20 bars (5 hours) of each session. Now
     resamples each session to a common length so the whole session counts.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _common import *

N_SESSIONS   = 30
SESSION_LEN  = 26      # resample every session to this many points
BAND         = 4       # Sakoe-Chiba radius; 0 would reduce DTW to Euclidean


def dtw_distance(a, b, band=BAND):
    """DTW with a Sakoe-Chiba band. O(n*band) instead of O(n^2)."""
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band); hi = min(m, i + band)
        for j in range(lo, hi + 1):
            c = (a[i - 1] - b[j - 1]) ** 2
            D[i, j] = c + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(np.sqrt(D[n, m]))


def resample_to(x, n):
    x = np.asarray(x, float)
    if len(x) == n: return x
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)


def main():
    df = fetch("GC=F", period="60d", interval="15m").dropna()
    df.index = to_athens(df.index)
    df["date"] = df.index.date

    raw = [g["close"].values for _, g in df.groupby("date") if len(g) >= SESSION_LEN]
    if len(raw) < 5:
        raise DataError(f"only {len(raw)} usable sessions")
    raw = raw[-N_SESSIONS:]                      # most RECENT, not first
    dates = sorted(set(df["date"]))[-len(raw):]

    # z-normalise each session: compares SHAPE, not level
    sessions = [znorm(resample_to(s, SESSION_LEN)) for s in raw]
    n = len(sessions)

    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(sessions[i], sessions[j])
            D[i, j] = D[j, i] = d

    # most and least typical sessions
    mean_d = D.sum(axis=1) / (n - 1)
    print(f"  sessions        : {n} (resampled to {SESSION_LEN} pts, z-normalised)")
    print(f"  most typical    : {dates[int(mean_d.argmin())]}  (mean DTW {mean_d.min():.2f})")
    print(f"  most unusual    : {dates[int(mean_d.argmax())]}  (mean DTW {mean_d.max():.2f})")
    iu = np.triu_indices(n, 1)
    k = int(np.argmin(D[iu]))
    print(f"  closest pair    : {dates[iu[0][k]]} <-> {dates[iu[1][k]]}  (DTW {D[iu][k]:.2f})")
    print(f"  yesterday vs all: mean DTW {mean_d[-1]:.2f}  "
          f"({'typical' if mean_d[-1] < np.median(mean_d) else 'unusual'} session)")

    plt.figure(figsize=(9, 7))
    plt.imshow(D, cmap="viridis", aspect="auto")
    plt.colorbar(label="DTW distance (z-normalised shape)")
    plt.title("Gold Intraday Session Shape — DTW Distance Matrix")
    plt.xlabel("Session (oldest → newest)"); plt.ylabel("Session (oldest → newest)")
    plt.savefig(OUT / "dtw_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    pd.DataFrame(D, index=dates, columns=dates).to_csv(OUT / "dtw_matrix.csv")
    print(f"  saved -> {OUT}/dtw_matrix.png, dtw_matrix.csv")


if __name__ == "__main__":
    main()
