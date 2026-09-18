# Patch guide — replace this with that

Install once:

```bash
# put goldlib/ next to your notebooks, then at the top of EVERY notebook:
import sys, subprocess
sys.path.insert(0, "/Users/elena_nael/Desktop/gold")
if subprocess.run([sys.executable, "validate.py"]).returncode != 0:
    raise SystemExit("Preflight failed — do not trade off this run.")

from goldlib import *
ctx = get_context()          # one fetch, every notebook, same numbers
print(ctx.summary())
```

---

## 1. Hardcoded price — `SUNDAY_CHECK`

**Delete:**
```python
GC_FUTURES_PRICE = 4725.0
OPTIONS_PROXY    = "GLD"
MULTIPLIER       = 10.0
print(f"[FETCH] Using configured GC price: ${GC_FUTURES_PRICE:.2f}")
```

**Replace with:**
```python
ctx = get_context()                    # raises if the fetch fails
GC_FUTURES_PRICE = ctx.gc              # keep the old name if other cells use it
MULTIPLIER       = ctx.ratio           # 10.9106, not 10.0
```

Same file, further down — **delete the entire `DAILY PRE-MARKET CHECK`
baseline block**:
```python
SUNDAY_BASE_GOLD = 4730.70
SUNDAY_BASE_VIX  = 17.19
```

**Replace with:**
```python
base = yf.download("GC=F", period="7d", interval="1d", progress=False)
SUNDAY_BASE_GOLD = float(base["Close"].dropna().iloc[0])   # actual week-ago bar
require_fresh(str(base.index[-1]), max_age_hours=96, label="GC baseline")
```

---

## 2. The ×10 bug — `SUNDAY_CHECK`, `Gold_Weekly_Prep`

**Delete every line of this shape:**
```python
implied_gold  = gld_spot * 10
gold_equiv    = strike * 10
gamma_walls_gold = [f"${s*10:,.0f}" for s in gamma_walls_gld]
print(f"Gold Futures : ~${gld_spot*10:.1f}  (GLD × 10 proxy)")
```

**Replace with:**
```python
implied_gold     = ctx.gld_to_gold(ctx.gld)
gold_equiv       = ctx.gld_to_gold(strike)
gamma_walls_gold = [f"${ctx.gld_to_gold(s):,.0f}" for s in gamma_walls_gld]
print(f"Gold Futures : ${ctx.gc:,.1f}  (ratio {ctx.ratio:.4f}x)")
```

What this changes in your output:

| GLD strike | ×10 said | Correct | Error |
|---|---|---|---|
| $370 | $3,700 | **$4,037** | 337 |
| $400 | $4,000 | **$4,364** | 364 |
| $410 | $4,100 | **$4,473** | 373 |
| $415 | $4,150 | **$4,528** | 378 |
| $420 | $4,200 | **$4,582** | 382 |
| $430 | $4,300 | **$4,692** | 392 |

---

## 3. Expiring manual fields — `Gold_Weekly_Prep`

**Delete:**
```python
NOTES = {
    "bias": "BEAR-RANGE — valid bearish setup (score -4)...",
    "trade_plan": "PRIMARY: Short rallies into 4680-4720...",
    "macro_theme": "Dual headwind: oil-driven inflation revival...",
}
```

**Replace with:**
```python
NOTES = {
    "bias":        manual_field("...", as_of="2026-08-16", label="bias"),
    "trade_plan":  manual_field("...", as_of="2026-08-16", label="trade_plan"),
    "macro_theme": manual_field("...", as_of="2026-08-16", label="macro_theme"),
}
for k, v in NOTES.items():
    print(f"  {k}: {v.get(strict=False)}")   # prints None + loud warning if expired
```

Set `strict=True` once you trust yourself to update them.

---

## 4. Six GEX flip levels — everywhere

**Delete every bespoke flip calculation.** They were all variations on
"find the strike where something is biggest", which is why one of them
returned `$205` (the lowest strike in the chain).

**Replace with:**
```python
chain = build_chain(ctx)          # cols: strike, side, oi, iv, dte, volume
chain["strike"] = ctx.gld_to_gold(chain["strike"])   # GLD -> gold, once

rep = compute_gex(chain, spot=ctx.gc, risk_free=ctx.risk_free)
print(rep.summary())
```

The flip is now found by re-evaluating gamma across a **spot grid** and
locating the zero crossing — which is what a flip level actually is.
If there is no crossing within ±35% of spot it returns `None` and says so,
instead of inventing a number.

---

## 5. Six put/call ratios

**Delete** each ad-hoc computation.

**Replace with the single definition:**
```python
pc = put_call_ratio(chain, spot=ctx.gc, basis="oi")
print(f"P/C (OI, ±10% strikes, ≤45 DTE): {pc:.3f}")
```

Always print the qualifier. `0.51` and `0.88` were both "right" — they were
measuring different things and neither said which.

---

## 6. Regime models that did not converge — `HSMM.py`, `FUNDAMENTALS`

**Delete:**
```python
warnings.filterwarnings('ignore')     # this is why you never saw the failures
model = HSMM(...); model.fit(X)
states = model.predict(X)
print("Inferred Market States:", states)
```

**Replace with:**
```python
res    = model.fit(X)
states = model.predict(X)
require_converged(res, label="HSMM", states=states)   # raises on 48/50 flat line
print("Inferred Market States:", states)
```

Your last run printed state `1` for 48 of 50 observations after logging
`Model is not converging`. That is not a regime classification. Either drop
to 2 states, lengthen the sample, or exclude the model from the weekly score —
but do not let it contribute a number.

---

## 7. Six-observation fits — `FUNDAMENTALS`

**Delete:**
```python
model = sm.OLS(y, X).fit()
print(model.summary())
mc = monte_carlo(returns, n=10000)
print(f"Probability Gold ↑: {mc.p_up:.1%}")
```

**Replace with:**
```python
require_min_obs(len(y), "ols", "gold AR(1)")            # floor 30
model = sm.OLS(y, X).fit()

require_min_obs(len(returns), "montecarlo", "gold MC")  # floor 250
mc = monte_carlo(returns, n=10000)
```

Also **delete the COT block that printed** `Non-Commercial Net: 2 contracts`
followed by `Extreme Non-Commercial Shorts → market may reverse up`. Add:

```python
if abs(noncomm_net) < 1000:
    raise ValueError(f"COT parse returned {noncomm_net} contracts — "
                     f"implausible for COMEX gold. Fix the parser.")
```

Keep the 650-observation regression. `AR(1) = −0.1019, p = 0.009` is real
and it is the best single finding in your whole stack: **hourly gold returns
mean-revert.** That is a tradable statement.

---

## 8. Kelly f* = 5.64 — `quant_behavioral_gld`

**Delete:**
```python
f_star = (mu - r) / sigma**2
print(f"GLD f* = {f_star:.2f}")
```

**Replace with:**
```python
f_star = (mu - r) / sigma**2
f_use  = cap_kelly(f_star, fraction=0.25, hard_cap=0.20, label="GLD Kelly")
print(f"raw f* = {f_star:.2f} (artefact) → using {f_use:.1%} of capital")
```

5.64 means 564% of capital. On a funded account that is a same-day breach.

---

## 9. The EV table — `calculations`

This is the one worth reading twice.

**Delete:**
```python
probs = [0.10, 0.15, 0.30, 0.25, 0.12, 0.08]     # hand-assigned, 45% up / 25% down
ev = sum(pnl * p for pnl, p in zip(pnls, probs))
print(f"EXPECTED VALUE = {ev:.1f}\nPOSITIVE EV ✅")
```

**Replace with:**
```python
tbl = ev_table(spot=ctx.gc, sigma=atm_iv, horizon_days=30,
               payoff_fn=lambda S: max(S - K, 0) * 100,
               cost=premium * 100,
               user_probs=[0.10, 0.15, 0.30, 0.25, 0.12, 0.08])
print(tbl.round(3).to_string(index=False))
print(summarise(tbl))
```

Run against your own ATM call, this prints:

```
EV at market-implied probabilities : $-1,144
EV at your probabilities           : $+3,358
Of which comes from your VIEW      : $+4,502
>> WARNING: structure is EV-negative. All the edge is your view.
```

Your "POSITIVE EV ✅ +4,036" was entirely your bullish probability vector.
The option itself is priced slightly against you. That does not mean don't
take it — it means the trade is a **directional view**, not a structural
edge, and it should be sized as one.

---

## 10. Seasonality — drift and timezone

**Delete:**
```python
df['quarter_hour'] = df.index.time
profile = df.groupby('quarter_hour')['Close'].mean()   # averages PRICE
```

**Replace with:**
```python
prof = intraday_return_profile(bars, tz="Europe/Athens", freq="15min")
prof[prof["significant"]]     # only windows with |t| > 2
```

Averaging price across a sample that ran 4,000 → 5,300 → 4,400 guarantees an
upward slope. You need cumulative return from each session's open, averaged
across sessions.

Then settle the timezone question empirically:
```python
verify_timezone(bars, event_utc="2026-08-12 12:30")   # July CPI release
```
Whichever timezone puts the biggest bar at 12:30 wall-clock is the one your
bars are actually in. If it comes back UTC, every intraday window in your
notes is off by three hours.

---

## 11. SGE premium — `Gold_Weekly_Prep`

`-$3,947/oz` is a CNY ETF price compared against USD/oz.

Until you have a real Au(T+D) feed, **remove it from the scorecard entirely**:

```python
SCORECARD.pop("SGE premium", None)     # broken input must not carry a vote
```

It is currently the only bearish line in a 16-row scorecard, so a broken
cell is moving your net score.

---

## 12. Parse-failure zeros — `SUNDAY_CHECK`

`Producer Net +0`, `Crude MM +0`, `Dealer Net +0` are failed parses being
scored as flat positioning.

**Add before scoring:**
```python
for name, val in positions.items():
    if val == 0:
        raise ValueError(f"{name} parsed as exactly 0 — almost certainly a "
                         f"column-name mismatch, not flat positioning.")
```

---

## 13. Which price is Friday's close

You have 4,380.40 (daily bar) and 4,437.30 (30-min bar) for the same session,
because `GC=F` daily on yfinance often reports the prior settlement.

**Pick one and declare it** at the top of `goldlib/prices.py`:
```python
SERIES = "front_intraday"    # 'front_intraday' | 'front_daily' | 'dec_daily'
```

Front-month intraday is the right default for what you trade. December futures
run about $30–40 above front — never mix them in one table.

---

## Order to do this in

1. `goldlib/` on the path, `validate.py` wired into every notebook — 20 min
2. Kill every `× 10` (patch 2) — biggest single P&L error
3. Kill the hardcoded price (patch 1) — this is the one that generated a
   fully-formed, completely wrong trade plan
4. Manual field expiry (patch 3)
5. Everything else

1–3 remove the errors that could actually cost you money. The rest is
research hygiene.
