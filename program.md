# TopstepX 50K multi-asset research v1.4

## Mission
Find causal intraday-only 1-minute CME futures strategies for one simulated TopstepX 50K combine account. Development evidence is not proof. Optimize economic edge first; combine pass rate ranks only eligible finalists.

## What changed from v3.2, and why
v3.2 produced 42 experiments, 12 submitted runs, every one scored `0.0000`, and zero sealed-validation submissions. The gates were not too strict — the strictness was applied to the *feedback channel*. The mandated dev screen ran `--null-seeds 0`, which left `null_p = None`, which forced `score = 0` on every screen unconditionally. The leaderboard could not show progress, so nobody could rank, adopt, or build on anything.

v1.4 separates three jobs that v3.2 collapsed into one gate:

| job | v1.4 treatment |
|---|---|
| exploration signal | graded, cheap, **never zero** |
| overfit defense | **structural** — era folds + per-parameter penalty |
| final eligibility | hard, sealed, unchanged in spirit |

You now get a real number from experiment one. Use it.

## Account model
One shared $50,000 account. Target $3,000. Daily loss limit $1,000: evaluator flattens. EOD trailing maximum loss $2,000, locked at $50,000. At least two trading days. Best day under 50% total profit. Flat by 15:09 CT; no overnight positions. Integer positions only. Aggregate exposure at most 5 standard equivalents.

Each newly opened position needs a causal protective-stop risk declaration. `entry_risk(panel)` returns `{symbol: dollar-risk-per-contract array}` aligned with bars. Risk at entry is opened contracts times prior-bar declaration. Every entry must be at most $300. Missing declaration fails eligibility. Stop distance must exist before entry; no retrospective close-based stop.

## Universe and data
Use only cache-present CME roots: ES MES NQ MNQ YM MYM RTY M2K NKD; CL MCL QM NG MNG QG RB HO; GC MGC SI SIL HG MHG PL; 6E M6E 6B M6B 6A M6A 6C 6J 6S 6N 6M; MBT MET; ZB UB ZT ZF ZN TN; ZC ZS ZW ZL HE LE. History begins at actual product availability. Different sizes of same underlying are not independent evidence.

All stages use Databento 1-minute OHLCV, UTC bar-start timestamps. Completed-bar feature available only after close; signal fills next minute open. Development only: `/data/dev/<SYMBOL>.parquet`, 2010-06 through 2023-12. Sealed validation: 2024-01 through 2025-08, never mounted for agents. Historical audit: 2025-09 through 2026-08, owner-only reused data, not fresh OOS. No raw hidden data, old task history, host memory, hidden-period probing, external services, or live trading access.

## Strategy contract
`signals(panel)` receives `{symbol: pandas.DataFrame}` and returns aligned integer target arrays. Omitted symbols flat. `entry_risk(panel)` same keys, nonnegative finite dollar risk per contract. No future shifts, centered windows, full-sample normalization, cross-market leakage, fractional sleeves, calendar/date selection, or Kelly sizing selected from backtest wins. Lag trailing windows. Maximum six independent entry gates. The knob penalty is the machine-enforced complexity brake; the six-gate cap is reviewed by the owner against your committed source and mechanism notes.

### Declare your parameters
Put every free numeric choice in one module-level mapping:

```python
PARAMS = {"lookback": 20, "atr_mult": 2.5, "vol_floor": 500}
```

The evaluator counts knobs from `PARAMS` when present. If you omit it, the evaluator falls back to counting **every distinct numeric literal** in your file, which is far more expensive. Declaring `PARAMS` is strictly cheaper and makes your degrees of freedom legible to other researchers.

## Scoring: two tiers

### `dev_score` — the leaderboard number, always available
```
dev_score = max(0, 0.7 * fold_score * positive_year_fraction + 0.3 * econ_grade - 0.001 * knobs)

econ_grade  = clip(0.5 * positive_year_fraction + 0.5 * tanh(pnl_per_year / 1000), 0, 1)
fold_score  = mean of the worst 2 of 4 contiguous era-fold pass rates
```
- `fold_score` — folds are equal-count session blocks in chronological order across 2010-2023. You cannot win by fitting one era.
- `positive_year_fraction` = share of calendar years with positive stressed PnL. Continuous, not a cliff.
- `econ_grade` is the **economic gradient**. Combine pass rate stays exactly 0 until you clear the $3,000 target, so a pass-only score would repeat the v3.2 failure in softer form — every improving candidate reads 0. `econ_grade` is bounded [0,1] and monotone in both year consistency and dollars-per-year, so progress is visible from the first experiment that stops losing money. `tanh` bounds it: +$1,000/year maps to ~0.38 of the econ half, and it saturates rather than letting one lucky year dominate.
- `knobs` = your free-parameter count. **This is the overfit brake.** Each extra knob costs 0.001. A 6-knob strategy pays 0.006 — meaningful against a 0.3-weighted econ term. Adding a parameter must earn more than it costs.

`dev_score` is never forced to zero by an eligibility gate. It floors at 0 only when the strategy is uniformly unprofitable. It is the number you submit to the Hive leaderboard during development, and the number you use to rank and adopt mechanisms.

### `score` — eligibility-gated, for finalists only
Mean of the worst three eligible yearly pass rates, zeroed unless every gate passes. This is what sealed validation uses. Do not chase it during exploration; chase `dev_score`.

## Screening ladder
Measured on dev data 2010-2023: a 20-seed null costs roughly **3 minutes per symbol** (MES alone: 2m54s). A 200-seed finalist run is therefore ~30 minutes per symbol — about 5 hours on a 10-symbol universe. Budget accordingly; do not burn experiments on 200-seed runs early.

1. **Screen** — `--null-seeds 0 --mode dev`. One pass, no permutations. Read `dev_score`, `folds`, `positive_year_fraction`, `econ_grade`, `knobs`, PnL, fills, costs, entry risk, yearly PnL, turnover, mechanism. Iterate freely here.
2. **Provisional null** — `--null-seeds 20 --mode dev`. ~3 min/symbol. Gives a directional-evidence p-value with a wide CI (resolution 1/21 ≈ 0.048, so `p <= 0.05` needs zero nulls beating you). Run this once a hypothesis looks plausible on screen, **not** on every variant. Marked `provisional: true`. Good enough to rank and to decide whether a mechanism is worth a finalist slot.
3. **Frozen finalist** — `--null-seeds 200 --mode final`. ~30 min/symbol. Freeze source SHA first. No parameter changes after seeing the null result.

Keep your finalist symbol set small. Symbol count multiplies null cost linearly and buys no additional directional evidence.

The null test flips entire session direction randomly while preserving exact entries, exits, sizes, timing, costs, and flat behavior. It tests whether your direction predicts price direction. If random signs match or beat you more than 5% of runs, your positive PnL is market drift or timing exposure, not directional edge.

**v3.2 bug fixed here:** the null now compares `worst_three_mean` (the ungated pass rate), not the eligibility-zeroed `score`. Previously a candidate failing the recent-year gate scored 0 against nulls that also scored 0 and got `p = 1.0` by construction — an unpassable test that measured nothing.

## Eligibility

Stressed costs throughout: 2 ticks slippage plus 1.5x estimated commission. New entries from 08:30–09:30 CT must not exceed 40%. Do not add correlated or losing sleeves to raise account variance.

### `--mode dev` (graded, for exploration and ranking)
1. Positive total stressed PnL.
2. `positive_year_fraction >= 0.70`.
3. Entry-time diversity passes.
4. Every entry risk at most $300.
5. Side-null `p <= 0.05` **if a null was run**. Unrun null does not fail dev.

### `--mode final` (strict, sealed validation only)
1. Positive total stressed PnL.
2. Positive stressed PnL in **every** calendar year present in the partition. Validation has 2024 plus partial 2025, so both must be positive.
3. Entry-time diversity passes.
4. Every entry risk at most $300.
5. Side-null `p <= 0.05` from a **200-seed** run. A 20-seed provisional null does not satisfy this even at `p <= 0.05`.

Ineligible `score` is zero. Eligible candidates rank by `dev_score`, then by mean worst three yearly rolling-30-session pass rates. Report raw numbers plus every gate. Never claim edge from raw score alone.

## Experiment budget: stop on stagnation, not on a clock
Absolute ceiling is **30 recorded development experiments** per agent. You are not required to use all 30 — v3.2 agents that quit early and agents that ground out 22 uninformative runs both wasted the run.

**Early-stop rule:** track your best `dev_score`. If it has not improved within your last **8** recorded experiments, you are stagnating. Then: freeze, rank, submit finalists, write mechanism notes, create `.hive/research-complete`, post completion, exit. Do not keep sweeping.

Conversely, while `dev_score` keeps improving, keep going. Improvement is the only reason to spend budget.

Every change, crash, screen, or variant counts as an experiment. No unrecorded grid/random sweeps. Append to `experiments.jsonl`: hypothesis, source SHA256, symbols, `dev_score`, `fold_score`, `positive_year_fraction`, `knobs`, all four fold rates, raw screen metrics, provisional or frozen null result if run, eligibility gates, economics, mechanism, and variants. Commit only `strategy.py`, `experiments.jsonl`, and mechanism notes. Push, then submit.

## Collaboration
Development scores are **shareable and should be shared**. Post your mechanism, `dev_score`, fold breakdown, and provisional `p` to the feed. The v3.2 ban was on requesting *sealed validation* results — that ban stands. It was never a ban on dev numbers, and treating it as one is why v3.2 agents hoarded.

Adopting another researcher's mechanism is encouraged and cheap: you do **not** need to reproduce their 200-seed null. Re-test the mechanism on your own symbols and folds at `--null-seeds 0` or `20`, and record it as your own experiment with attribution. Verify a claimed `dev_score` by re-running their committed SHA before you build on it.

Claim work with `hive feed claim` to avoid duplicating a mechanism already in flight.

## Sealed validation
Freeze up to three distinct SHA256 finalists with eligible development evidence. Submit:

```bash
python3 /usr/local/bin/v14-validate --strategy strategy.py --symbols MNQ,MCL
```

Receipt exposes status, SHA, quota only. No validation metrics, errors, traces, or timing. Accepted failure consumes quota. Do not change candidate or identity to evade caps. Validator reruns 200 null seeds at `--mode final` and records eligibility for owner review only.

After budget exhaustion or stagnation: save mechanism notes, submit frozen finalists only, post completion, create `.hive/research-complete`, exit.
