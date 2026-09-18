"""
_common.py — shared helpers for the GC toolkit.
Import at the top of every script: from _common import *
"""
import os, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

class DataError(RuntimeError): pass

def last(df, col="Close"):
    """float() on a 1-element Series is deprecated; also flattens yfinance MultiIndex."""
    s = df[col] if col in df else df
    if isinstance(s, pd.DataFrame): s = s.iloc[:, 0]
    a = np.asarray(s.dropna()).ravel()
    if a.size == 0: raise DataError(f"empty {col}")
    return float(a[-1])

def flatten(df):
    """yfinance MultiIndex -> flat lowercase columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.columns = [str(c).lower() for c in df.columns]
    return df

def fetch(ticker="GC=F", **kw):
    import yfinance as yf
    df = yf.download(ticker, progress=False, auto_adjust=False, **kw)
    if df is None or df.empty:
        raise DataError(f"{ticker}: no data returned for {kw}")
    return flatten(df)

def to_athens(idx):
    """
    yfinance returns GC=F intraday tz-aware in exchange time. Only localize
    if genuinely naive — assuming NY on an already-aware index is a silent
    3-hour error.
    """
    if idx.tz is None:
        idx = idx.tz_localize("America/New_York")
    return idx.tz_convert("Europe/Athens")

def znorm(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 1e-12 else x - x.mean()
