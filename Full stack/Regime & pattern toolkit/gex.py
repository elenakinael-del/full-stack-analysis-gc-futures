"""
goldlib.gex
===========
ONE gamma-exposure calculation. Your stack currently prints six different
flip levels in a single session ($205, $3310, $3550, $3562, $3813, $4266)
because six cells each invented their own.

The flip level is NOT a strike. It is the spot price at which net dealer
gamma crosses zero. You find it by RE-EVALUATING gamma across a grid of
candidate spot prices and locating the zero crossing. Picking the strike
where something looks big is what produced the $205 answer (that was just
the lowest strike in the chain).

Sign convention is an ASSUMPTION, not a fact. Declared explicitly below
and switchable, so at least it is consistent everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm


# Standard dealer assumption: customers are net long puts and net short calls,
# so dealers are net long calls / short puts -> calls +gamma, puts -gamma.
DEALER_LONG_CALLS = True
CONTRACT_MULTIPLIER = 100.0


def bs_gamma(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1.0 / 365 / 24)          # floor at 1 hour
    sigma = np.clip(sigma, 0.01, 5.0)          # kill the 0.0000001 IV rows
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_delta(S, K, T, r, sigma, q=0.0, call=True):
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    T = np.maximum(T, 1.0 / 365 / 24)
    sigma = np.clip(sigma, 0.01, 5.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if call:
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


@dataclass
class GexReport:
    spot: float
    net_gex: float
    flip: Optional[float]
    regime: str
    call_wall: Optional[float]
    put_wall: Optional[float]
    max_pain: Optional[float]
    curve: pd.DataFrame        # spot grid -> net gex
    by_strike: pd.DataFrame
    units: str

    def summary(self) -> str:
        f = f"${self.flip:,.0f}" if self.flip is not None else "none in grid"
        return (
            f"  Spot          : ${self.spot:,.2f}\n"
            f"  Net GEX       : {self.net_gex/1e6:+,.1f}M {self.units}\n"
            f"  Zero-gamma flip: {f}\n"
            f"  Regime        : {self.regime}\n"
            f"  Call wall     : ${self.call_wall:,.0f}\n"
            f"  Put wall      : ${self.put_wall:,.0f}\n"
            f"  Max pain      : ${self.max_pain:,.0f}\n"
        )


def compute_gex(chain: pd.DataFrame,
                spot: float,
                risk_free: float = 0.037,
                strike_window: float = 0.15,
                max_dte: int = 45,
                grid_pct: float = 0.35,
                grid_n: int = 401,
                dealer_long_calls: bool = DEALER_LONG_CALLS) -> GexReport:
    """
    Parameters
    ----------
    chain : DataFrame with columns
        ['strike', 'side', 'oi', 'iv', 'dte']
        strike and spot must be in the SAME units. Convert GLD strikes with
        ctx.gld_to_gold() BEFORE calling, or pass everything in GLD terms.
    strike_window : keep strikes within +/- this fraction of spot
    max_dte : ignore anything longer dated; LEAPS swamp the near-term picture
    grid_pct : how far either side of spot to search for the flip

    Returns GexReport. `flip` is None if no zero crossing exists in the grid,
    which is an honest answer and much better than reporting the lowest strike.
    """
    df = chain.copy()
    need = {"strike", "side", "oi", "iv", "dte"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"chain missing columns: {sorted(missing)}")

    df["side"] = df["side"].str.lower()
    df = df[df["side"].isin(["call", "put"])]
    df = df[(df["dte"] > 0) & (df["dte"] <= max_dte)]
    df = df[df["oi"].fillna(0) > 0]
    df = df[(df["iv"] > 0.01) & (df["iv"] < 5.0)]
    lo, hi = spot * (1 - strike_window), spot * (1 + strike_window)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]

    if df.empty:
        raise ValueError("no usable option rows after filtering")

    sign = np.where(df["side"].eq("call"), 1.0, -1.0)
    if not dealer_long_calls:
        sign = -sign

    T = df["dte"].to_numpy() / 365.0

    # ---- net GEX at current spot -----------------------------------------
    g = bs_gamma(spot, df["strike"].to_numpy(), T, risk_free, df["iv"].to_numpy())
    gex_rows = sign * g * df["oi"].to_numpy() * CONTRACT_MULTIPLIER * spot ** 2 * 0.01
    net_gex = float(np.nansum(gex_rows))

    by_strike = (df.assign(gex=gex_rows)
                   .groupby(["strike", "side"], as_index=False)
                   .agg(gex=("gex", "sum"), oi=("oi", "sum")))

    # ---- flip: re-evaluate gamma across a SPOT GRID -----------------------
    grid = np.linspace(spot * (1 - grid_pct), spot * (1 + grid_pct), grid_n)
    curve = np.empty_like(grid)
    K = df["strike"].to_numpy()
    OI = df["oi"].to_numpy()
    IV = df["iv"].to_numpy()
    for i, S in enumerate(grid):
        gi = bs_gamma(S, K, T, risk_free, IV)
        curve[i] = np.nansum(sign * gi * OI * CONTRACT_MULTIPLIER * S ** 2 * 0.01)

    flip = None
    crossings = np.where(np.sign(curve[:-1]) != np.sign(curve[1:]))[0]
    if len(crossings):
        # take the crossing nearest to spot
        cand = []
        for i in crossings:
            x0, x1, y0, y1 = grid[i], grid[i + 1], curve[i], curve[i + 1]
            if y1 != y0:
                cand.append(x0 - y0 * (x1 - x0) / (y1 - y0))
        if cand:
            flip = float(min(cand, key=lambda x: abs(x - spot)))

    if flip is None:
        regime = ("LONG GAMMA (no flip within +/-%.0f%% of spot — moves suppressed "
                  "across the whole searched range)" % (grid_pct * 100)
                  if net_gex > 0 else "SHORT GAMMA (no flip in range)")
    elif net_gex > 0:
        regime = "LONG GAMMA — dealers dampen moves; fade extremes"
    else:
        regime = "SHORT GAMMA — dealers amplify moves; respect breakouts"

    # ---- walls: largest absolute gamma above / below spot -----------------
    calls = by_strike[(by_strike["side"] == "call") & (by_strike["strike"] > spot)]
    puts = by_strike[(by_strike["side"] == "put") & (by_strike["strike"] < spot)]
    call_wall = float(calls.loc[calls["gex"].abs().idxmax(), "strike"]) if len(calls) else np.nan
    put_wall = float(puts.loc[puts["gex"].abs().idxmax(), "strike"]) if len(puts) else np.nan

    # ---- max pain: strike minimising total payout to option HOLDERS -------
    strikes = np.sort(df["strike"].unique())
    call_oi = df[df.side == "call"].groupby("strike")["oi"].sum()
    put_oi = df[df.side == "put"].groupby("strike")["oi"].sum()
    pain = []
    for S in strikes:
        c = float((np.maximum(S - call_oi.index.to_numpy(), 0) * call_oi.to_numpy()).sum())
        p = float((np.maximum(put_oi.index.to_numpy() - S, 0) * put_oi.to_numpy()).sum())
        pain.append(c + p)
    max_pain = float(strikes[int(np.argmin(pain))])

    return GexReport(
        spot=spot, net_gex=net_gex, flip=flip, regime=regime,
        call_wall=call_wall, put_wall=put_wall, max_pain=max_pain,
        curve=pd.DataFrame({"spot": grid, "net_gex": curve}),
        by_strike=by_strike.sort_values("strike"),
        units="$ per 1% move",
    )


# --------------------------------------------------------------------------
# Canonical put/call ratio — you currently print six different values
# --------------------------------------------------------------------------

def put_call_ratio(chain: pd.DataFrame, spot: float, basis: str = "oi",
                   strike_window: float = 0.10, max_dte: int = 45) -> float:
    """
    ONE definition, used everywhere:
        basis    : 'oi' or 'volume'
        strikes  : within +/-10% of spot
        expiries : <= 45 DTE
    Report it as "P/C (OI, +/-10%, <=45d)" so the number is never ambiguous.
    """
    col = {"oi": "oi", "volume": "volume"}[basis]
    df = chain.copy()
    df["side"] = df["side"].str.lower()
    lo, hi = spot * (1 - strike_window), spot * (1 + strike_window)
    df = df[(df["strike"] >= lo) & (df["strike"] <= hi)]
    df = df[(df["dte"] > 0) & (df["dte"] <= max_dte)]
    calls = df.loc[df.side == "call", col].fillna(0).sum()
    puts = df.loc[df.side == "put", col].fillna(0).sum()
    if calls <= 0:
        return float("nan")
    return float(puts / calls)
