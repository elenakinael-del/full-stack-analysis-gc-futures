"""
goldlib.seasonality
===================
Fixes two bugs:

1. Averaging raw PRICE across a trending sample. On a series that went
   4,000 -> 5,300 -> 4,400 the "profile" mostly shows the trend, and the
   upward slope from 06:00 to 17:00 is guaranteed by construction.
   Fix: average CUMULATIVE RETURN FROM SESSION OPEN, per session, then
   average across sessions.

2. Timezone. One of your charts is labelled Athens; its volatility peaks
   at 08:30-10:15 and 20:00-21:30 read like UTC (London spot open and
   COMEX settle). verify_timezone() settles it against a known event.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .guards import require_min_obs


SESSION_TZ = "Europe/Athens"      # declare it once, use it everywhere


def _ensure_tz(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    out = df.copy()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    out.index = out.index.tz_convert(tz)
    return out


def intraday_return_profile(bars: pd.DataFrame,
                            tz: str = SESSION_TZ,
                            price_col: str = "Close",
                            session_start: str = "01:00",
                            freq: str = "15min") -> pd.DataFrame:
    """
    Drift-free intraday seasonality.

    Returns a frame indexed by time-of-day with:
        mean_cum_ret : average cumulative % move from session open
        mean_abs_ret : average absolute bar return (the volatility clock)
        n_sessions   : how many sessions contributed

    REPLACES: groupby(quarter_hour)['Close'].mean()
    """
    df = _ensure_tz(bars, tz)
    df = df[[price_col]].dropna()

    df["session"] = (df.index - pd.Timedelta(session_start + ":00")).date
    df["tod"] = df.index.floor(freq).time

    df["ret"] = df.groupby("session")[price_col].pct_change()

    # cumulative return from that session's first bar
    first = df.groupby("session")[price_col].transform("first")
    df["cum_ret"] = df[price_col] / first - 1.0

    n_sessions = df["session"].nunique()
    require_min_obs(len(df), "seasonality", "intraday profile")

    prof = (df.groupby("tod")
              .agg(mean_cum_ret=("cum_ret", "mean"),
                   se_cum_ret=("cum_ret", "sem"),
                   mean_abs_ret=("ret", lambda s: s.abs().mean()),
                   n=("ret", "count"))
              .assign(n_sessions=n_sessions))

    # t-stat on the drift: is any window actually significant?
    prof["t_stat"] = prof["mean_cum_ret"] / prof["se_cum_ret"].replace(0, np.nan)
    prof["significant"] = prof["t_stat"].abs() > 2.0
    return prof


def verify_timezone(bars: pd.DataFrame, event_utc: str,
                    candidate_tzs=("UTC", "Europe/Athens", "America/New_York"),
                    window_min: int = 30) -> pd.DataFrame:
    """
    Settle the Athens-vs-UTC question empirically.

    Pass a timestamp you KNOW moved the market, in UTC — e.g. the July CPI
    print, 2026-08-12 12:30 UTC. This locates the highest-|return| bar near
    that moment and tells you what wall-clock label it carries in each
    candidate timezone. The tz whose label matches the true release time is
    the one your bars are in.

    Example
    -------
    verify_timezone(bars, "2026-08-12 12:30")
    """
    ev = pd.Timestamp(event_utc, tz="UTC")
    out = []
    for tz in candidate_tzs:
        d = _ensure_tz(bars, tz)
        r = d["Close"].pct_change().abs()
        lo = (ev - pd.Timedelta(minutes=window_min)).tz_convert(tz)
        hi = (ev + pd.Timedelta(minutes=window_min)).tz_convert(tz)
        seg = r.loc[lo:hi]
        if seg.empty:
            out.append({"tz": tz, "peak_bar": None, "abs_ret": np.nan})
            continue
        out.append({"tz": tz,
                    "peak_bar": str(seg.idxmax().time()),
                    "abs_ret": float(seg.max())})
    return pd.DataFrame(out)
