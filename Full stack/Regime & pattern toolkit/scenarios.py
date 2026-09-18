"""
goldlib.scenarios
=================
Fixes the EV table.

Your version hand-assigns probabilities 10/15/30/25/12/8 — 45% up, 25% down
— and then reports "POSITIVE EV". With a bullish probability vector hardcoded
in, the table cannot print anything else. It is your prior restated as
arithmetic, not evidence.

Fix: derive the probabilities from the option market itself. The lognormal
implied by ATM IV is the market's own distribution. If EV is still positive
against THAT, you have found something. If it isn't, you have saved yourself
a trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def implied_scenario_probs(spot: float, sigma: float, horizon_days: float,
                           moves=(-0.10, -0.05, 0.0, 0.05, 0.10, 0.15),
                           r: float = 0.0) -> pd.DataFrame:
    """
    Probability of landing in each bucket under the market-implied lognormal.

    Buckets are the midpoints between consecutive `moves`, so they partition
    the real line and the probabilities sum to 1 by construction.
    """
    T = horizon_days / 365.0
    vol = sigma * np.sqrt(T)
    mu = (r - 0.5 * sigma ** 2) * T

    levels = np.array([spot * (1 + m) for m in moves])
    edges = np.concatenate([[-np.inf],
                            (levels[:-1] + levels[1:]) / 2.0,
                            [np.inf]])

    probs = []
    for i in range(len(levels)):
        lo, hi = edges[i], edges[i + 1]
        z_lo = -np.inf if lo == -np.inf else (np.log(lo / spot) - mu) / vol
        z_hi = np.inf if hi == np.inf else (np.log(hi / spot) - mu) / vol
        probs.append(norm.cdf(z_hi) - norm.cdf(z_lo))

    return pd.DataFrame({
        "move": moves,
        "spot": levels,
        "prob_implied": probs,
    })


def ev_table(spot: float, sigma: float, horizon_days: float,
             payoff_fn, cost: float,
             moves=(-0.10, -0.05, 0.0, 0.05, 0.10, 0.15),
             user_probs=None) -> pd.DataFrame:
    """
    payoff_fn : callable(terminal_spot) -> gross payoff in $
    cost      : premium paid, in $

    If you pass `user_probs`, the table shows BOTH columns side by side so
    you can see exactly how much of your positive EV comes from your view
    rather than from the structure. That difference is the trade thesis —
    it should be stated out loud, not buried in a probability vector.
    """
    df = implied_scenario_probs(spot, sigma, horizon_days, moves)
    df["pnl"] = [payoff_fn(s) - cost for s in df["spot"]]
    df["ev_implied"] = df["pnl"] * df["prob_implied"]

    if user_probs is not None:
        up = np.asarray(user_probs, dtype=float)
        if not np.isclose(up.sum(), 1.0, atol=1e-6):
            raise ValueError(f"user_probs sum to {up.sum():.4f}, must sum to 1.0")
        df["prob_user"] = up
        df["ev_user"] = df["pnl"] * up
        df["edge_from_view"] = df["ev_user"] - df["ev_implied"]

    return df


def summarise(df: pd.DataFrame) -> str:
    ev_i = df["ev_implied"].sum()
    lines = [f"  EV at market-implied probabilities : ${ev_i:+,.0f}"]
    if "ev_user" in df:
        ev_u = df["ev_user"].sum()
        lines += [
            f"  EV at your probabilities           : ${ev_u:+,.0f}",
            f"  Of which comes from your VIEW      : ${ev_u - ev_i:+,.0f}",
        ]
        if ev_i <= 0 < ev_u:
            lines.append("  >> WARNING: structure is EV-negative. All the edge is your view.")
    lines.append("  VERDICT: " + ("POSITIVE EV vs market" if ev_i > 0
                                  else "NEGATIVE EV vs market — the option is fairly/richly priced"))
    return "\n".join(lines)
