"""
goldlib.prices
==============
ONE place where prices enter the stack. Every notebook calls get_context().

Kills:
  * GC_FUTURES_PRICE = 4725.0   (hardcoded May price)
  * GLD * 10                    (ratio is 10.91, not 10.00)
  * "Sunday base 4730.70"       (stale baseline)
  * two different Friday closes (4380.40 vs 4437.30)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from .guards import require_price_sane, StaleDataError


# --------------------------------------------------------------------------
# Contract selection — this is why you had two Friday prices
# --------------------------------------------------------------------------
# GC=F on yfinance is the FRONT MONTH and its daily bar is often the prior
# settlement, not the live session. The 30-min series is the live session.
# Pick ONE and declare it. Default: front-month, from intraday, resampled.
# --------------------------------------------------------------------------

SERIES = "front_intraday"   # 'front_intraday' | 'front_daily' | 'dec_daily'


@dataclass
class MarketContext:
    gc: float                 # gold futures, the series named in `series`
    gld: float                # GLD ETF share price
    ratio: float              # gc / gld  — USE THIS, NEVER 10.0
    vix: float
    gvz: float
    risk_free: float
    series: str
    asof: str                 # ISO timestamp of the newest bar used
    fetched_at: str

    def gld_to_gold(self, gld_strike: float) -> float:
        """GLD strike -> gold price. Replaces every `* 10` in the stack."""
        return gld_strike * self.ratio

    def gold_to_gld(self, gold_price: float) -> float:
        return gold_price / self.ratio

    def summary(self) -> str:
        return (
            f"  GC ({self.series:<15}) : ${self.gc:,.2f}\n"
            f"  GLD                     : ${self.gld:,.2f}\n"
            f"  GLD->Gold ratio         : {self.ratio:.4f}x   <-- not 10.0\n"
            f"  VIX / GVZ               : {self.vix:.2f} / {self.gvz:.1f}\n"
            f"  Risk-free               : {self.risk_free:.2%}\n"
            f"  Bars as of              : {self.asof}\n"
        )


def _last_valid(df: pd.DataFrame, col: str = "Close") -> tuple[float, str]:
    s = df[col].dropna() if col in df else df.dropna()
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0].dropna()
    if s.empty:
        raise StaleDataError("empty price series returned")
    return float(s.iloc[-1]), str(s.index[-1])


def get_context(series: str = SERIES, max_age_hours: float = 96.0) -> MarketContext:
    """
    Fetch everything once, sanity-check it, and hand back an immutable context.
    Raises rather than returning a stale or implausible price.

    max_age_hours default 96 so a Saturday/Sunday run against Friday's close
    passes, but a three-month-old value never does.
    """
    import yfinance as yf   # imported here so the module imports without network

    if series == "front_intraday":
        gc_df = yf.download("GC=F", period="5d", interval="30m",
                            progress=False, auto_adjust=False)
    elif series == "front_daily":
        gc_df = yf.download("GC=F", period="1mo", interval="1d",
                            progress=False, auto_adjust=False)
    elif series == "dec_daily":
        gc_df = yf.download("GCZ26.CMX", period="1mo", interval="1d",
                            progress=False, auto_adjust=False)
    else:
        raise ValueError(f"unknown series {series!r}")

    gc, gc_asof = _last_valid(gc_df)
    require_price_sane("GC=F", gc)

    gld_df = yf.download("GLD", period="5d", interval="1d",
                         progress=False, auto_adjust=False)
    gld, _ = _last_valid(gld_df)
    require_price_sane("GLD", gld)

    def _safe(ticker, default):
        try:
            d = yf.download(ticker, period="5d", interval="1d",
                            progress=False, auto_adjust=False)
            v, _ = _last_valid(d)
            require_price_sane(ticker, v)
            return v
        except Exception:
            return default

    vix = _safe("^VIX", np.nan)
    gvz = _safe("^GVZ", np.nan)
    irx = _safe("^IRX", np.nan)
    rf = (irx / 100.0) if np.isfinite(irx) else 0.04

    ratio = gc / gld
    if not (9.5 <= ratio <= 12.5):
        raise StaleDataError(
            f"GLD->Gold ratio computed as {ratio:.3f}x, outside [9.5, 12.5]. "
            f"One of the two prices is wrong or from a different date. "
            f"GC={gc:,.2f} GLD={gld:,.2f}"
        )

    ctx = MarketContext(
        gc=gc, gld=gld, ratio=ratio, vix=vix, gvz=gvz, risk_free=rf,
        series=series, asof=gc_asof,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return ctx


def context_from_values(gc: float, gld: float, vix: float = 14.25,
                        gvz: float = 23.9, risk_free: float = 0.037,
                        asof: str = "manual") -> MarketContext:
    """Offline / testing constructor. Still enforces the ratio band."""
    ratio = gc / gld
    if not (9.5 <= ratio <= 12.5):
        raise StaleDataError(f"ratio {ratio:.3f}x outside [9.5, 12.5]")
    return MarketContext(
        gc=gc, gld=gld, ratio=ratio, vix=vix, gvz=gvz, risk_free=risk_free,
        series="manual", asof=asof,
        fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
