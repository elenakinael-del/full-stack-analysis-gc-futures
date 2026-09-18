# GC Futures Systematic Research Stack

Independent quantitative research on COMEX Gold Futures (GC=F).

## Research question
Can a systematic combination of regime detection, options flow,
COT positioning, and intraday liquidity structure produce consistent
edge in COMEX gold?

## Stack overview
[paste the workflow table above]

## Key findings
- GARCH: current vol at 88th percentile (26y sample), half-life 59 days
- HSMM: 3-state model, current regime volatile-neutral
- Seasonality: volatility clock confirmed (NY open 16:30 Athens);
  directional drift not statistically significant
- Template matching: V-shape pattern appears at 2.5% frequency
  (below the 8.1% noise floor — no structural edge)

## Environment
Python 3.13 / Anaconda · yfinance · arch · stumpy · hmmlearn
