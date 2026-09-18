"""
goldlib.guards
==============
Fail-loud guards. Every one of these raises rather than returning a
degraded value, because the failure mode in your stack was *broken input,
confident output*.

Import at the top of every notebook:

    from goldlib.guards import (
        StaleDataError, ModelError, SampleError,
        require_fresh, require_min_obs, require_converged,
        require_price_sane, cap_kelly, manual_field,
    )
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np


class StaleDataError(RuntimeError):
    """Raised when a value is older than its allowed shelf life."""


class ModelError(RuntimeError):
    """Raised when a fitted model did not converge."""


class SampleError(RuntimeError):
    """Raised when a fit has too few observations to mean anything."""


class PriceSanityError(RuntimeError):
    """Raised when a price is outside a plausible band."""


# --------------------------------------------------------------------------
# 1. Staleness
# --------------------------------------------------------------------------

def require_fresh(ts, max_age_hours: float, label: str = "value"):
    """
    Raise if `ts` is older than `max_age_hours`.

    Replaces: hardcoded GC_FUTURES_PRICE, and manual dict fields that
    silently survived from May into August.
    """
    if ts is None:
        raise StaleDataError(f"{label}: no timestamp recorded — treat as stale.")

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    age = datetime.now(timezone.utc) - ts
    if age > timedelta(hours=max_age_hours):
        raise StaleDataError(
            f"{label} is {age.total_seconds()/3600:.1f}h old "
            f"(limit {max_age_hours}h). Refresh it before using this cell."
        )
    return True


@dataclass
class ManualField:
    """A hand-typed value that carries its own expiry."""
    value: Any
    as_of: str          # ISO date, e.g. "2026-08-15"
    shelf_life_days: int = 7
    label: str = "manual field"

    def get(self, strict: bool = True):
        as_of = datetime.fromisoformat(self.as_of).replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - as_of).days
        if age_days > self.shelf_life_days:
            msg = (f"MANUAL FIELD EXPIRED — '{self.label}' written {self.as_of} "
                   f"({age_days}d ago, shelf life {self.shelf_life_days}d). "
                   f"Value NOT used.")
            if strict:
                raise StaleDataError(msg)
            warnings.warn(msg)
            return None
        return self.value


def manual_field(value, as_of, shelf_life_days=7, label="manual field"):
    return ManualField(value, as_of, shelf_life_days, label)


# --------------------------------------------------------------------------
# 2. Sample size
# --------------------------------------------------------------------------

MIN_OBS = {
    "ols": 30,
    "garch": 250,
    "hmm": 500,
    "hsmm": 500,
    "montecarlo": 250,
    "correlation": 60,
    "seasonality": 500,
}


def require_min_obs(n: int, kind: str, label: str = ""):
    """
    Replaces: OLS on 6 observations, Monte Carlo on 6 observations,
    "Probability Gold up: 100.0%".
    """
    floor = MIN_OBS.get(kind, 30)
    if n < floor:
        raise SampleError(
            f"{label or kind}: {n} observations, need >= {floor}. "
            f"A fit this small has no information — do not report it."
        )
    return True


# --------------------------------------------------------------------------
# 3. Convergence
# --------------------------------------------------------------------------

def require_converged(result, label: str = "model", min_distinct_states: int = 2,
                      states: Optional[np.ndarray] = None,
                      min_minority_share: float = 0.05):
    """
    Replaces: the HSMM that printed state 1 for 48/50 obs, and the
    Markov-switching fit that raised 'Invalid regime transition probabilities'.

    Checks, in order:
      1. statsmodels/hmmlearn convergence flag
      2. that the state sequence actually has >1 state
      3. that the minority state is not vanishingly rare (a 96/4 split is
         an outlier detector, not a regime model)
    """
    converged = None
    for attr in ("converged", "mle_retvals", "monitor_"):
        obj = getattr(result, attr, None)
        if obj is None:
            continue
        if attr == "converged":
            converged = bool(obj)
        elif attr == "mle_retvals" and isinstance(obj, dict):
            converged = bool(obj.get("converged", True))
        elif attr == "monitor_":
            converged = bool(getattr(obj, "converged", True))
        break

    if converged is False:
        raise ModelError(
            f"{label}: optimiser did not converge. Output is not a regime "
            f"classification. Fix the fit or drop the model from this week's weights."
        )

    if states is not None:
        states = np.asarray(states)
        uniq, counts = np.unique(states, return_counts=True)
        if len(uniq) < min_distinct_states:
            raise ModelError(
                f"{label}: only {len(uniq)} distinct state(s) inferred "
                f"({uniq.tolist()}). That is a flat line, not a regime model."
            )
        minority = counts.min() / counts.sum()
        if minority < min_minority_share:
            raise ModelError(
                f"{label}: minority state is {minority:.1%} of the sample "
                f"(need >= {min_minority_share:.0%}). This is detecting outliers, "
                f"not regimes. Reduce n_states or lengthen the sample."
            )
    return True


# --------------------------------------------------------------------------
# 4. Price sanity
# --------------------------------------------------------------------------

PRICE_BANDS = {
    "GC=F":  (1_000, 12_000),
    "GLD":   (100, 1_200),
    "^VIX":  (5, 90),
    "^GVZ":  (8, 90),
    "DX-Y.NYB": (70, 130),
    "^TNX":  (0.2, 12.0),
}


def require_price_sane(ticker: str, price: float):
    lo, hi = PRICE_BANDS.get(ticker, (None, None))
    if lo is None:
        return True
    if not (lo <= price <= hi):
        raise PriceSanityError(
            f"{ticker} = {price:,.2f} is outside the plausible band "
            f"[{lo:,}, {hi:,}]. Almost always a bad fetch or wrong column."
        )
    return True


# --------------------------------------------------------------------------
# 5. Kelly
# --------------------------------------------------------------------------

def cap_kelly(f_star: float, fraction: float = 0.25, hard_cap: float = 0.20,
              label: str = "Kelly") -> float:
    """
    Replaces: f* = 5.64 (564% of capital).

    Full Kelly on a 2-year sample is a sampling artefact, not a position size.
    Returns fractional Kelly, then clamps to a hard cap.
    """
    if not np.isfinite(f_star):
        raise ValueError(f"{label}: f* is not finite.")
    frac = f_star * fraction
    capped = float(np.clip(frac, -hard_cap, hard_cap))
    if abs(f_star) > 1.0:
        warnings.warn(
            f"{label}: raw f*={f_star:.2f} implies {f_star*100:.0f}% of capital. "
            f"This is a small-sample artefact. Using {capped*100:.1f}% "
            f"({fraction:.0%} Kelly, hard cap {hard_cap:.0%})."
        )
    return capped
