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


gc-futures-research/
│
├── README.md                          ← the workflow above, written up
│
├── 1_weekly_macro/
│   ├── FUNDAMENTALS.ipynb
│   ├── Daily_systematic_gold_macro_model.ipynb
│   ├── Gold_Williams_Setup_Scorer_3.ipynb
│   ├── gold_smart_money.ipynb
│   └── quant_behavioral_gld.ipynb
│
├── 2_weekly_options_levels/
│   ├── calculations.ipynb
│   ├── Gold_Weekly_Prep_v3.ipynb
│   └── SUNDAY_CHECK_fixed_1.ipynb
│
├── 3_daily_execution/
│   ├── DAILY_UPDATE.ipynb
│   ├── Gold_Morning_Bias_Engine.ipynb
│   └── NY_GOLD_LIQUIDITY_ENGINE.ipynb
│
├── 4_pattern_regime_toolkit/
│   ├── run_all.py                     ← entry point
│   ├── _common.py
│   ├── garch.py
│   ├── HSMM.py
│   ├── dtw.py
│   ├── matrix.py
│   ├── quarter.py
│   ├── template.py
│   ├── gameplan.py
│   └── gameplan2.py
│
└── outputs/                           ← add to .gitignore
