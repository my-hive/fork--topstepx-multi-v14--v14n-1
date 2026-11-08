"""Trusted multiasset 1-minute evaluator. Candidate supplies only signals(panel).

OHLC cannot resolve cross-market event order. Account adverse valuation sums each
held market's adverse extreme in the same UTC minute (even if extrema were not
simultaneous). DLL exits at those adverse prices plus costs, NOT exactly -$1000.
Gaps/rolls flatten at prior close plus costs; no synthetic forward-filled bars.
Daily reference ledger resets account each session. Rolling attempts consume its
PnL/adverse minima; MLL failure terminates attempt before subsequent reference
fills. Ledger is reference execution, not a claimed funded-account lifecycle.
EOD MLL floor = min(50000, highest EOD balance - 2000); strict best-day <50%.
v1.4 two-tier scoring. dev_score (graded, never zero) = mean of the worst 2 of 4
contiguous era-fold pass rates * positive-year fraction - 0.001 per free parameter.
That is the leaderboard number and it is available from experiment one. score
(eligibility-gated) stays strict and is the only number that matters for sealed
validation: mode=dev requires >=70% positive years, mode=final requires every
partition year positive plus a 200-seed side-null p<=0.05.
Data loader reads only selected symbols' dev.parquet (never sealed files).
This process is NOT a sandbox: candidate Python must run inside external sandbox.
"""
import argparse
import ast
import hashlib
import heapq
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .instruments import SPECS, MAX_EXPOSURE
except ImportError:
    from instruments import SPECS, MAX_EXPOSURE

START, TARGET, DLL, MLL, WINDOW = 50000., 3000., 1000., 2000., 30
FLAT_SEC = 15 * 3600 + 10 * 60
MINUTE = 60_000_000_000

# v1.4 graded development scoring. N_FOLDS contiguous era folds; dev score is the
# mean of the worst DEV_WORST_FOLDS fold pass rates, scaled by year consistency,
# minus a BIC-style penalty per free parameter. Never forced to zero, so the
# leaderboard carries a gradient. Hard gates apply only at mode='final'.
N_FOLDS, DEV_WORST_FOLDS = 4, 2
KNOB_PENALTY = 0.001
DEV_YEAR_FLOOR = 0.70
NULL_SEEDS_ALLOWED = (0, 20, 200)
# dev_score blends the fold-robust pass rate with a bounded economic grade so that
# candidates improve visibly before they clear the combine pass threshold.
DEV_PASS_WEIGHT = 0.7
ECON_PNL_SCALE = 1000.  # dollars/year that maps to tanh(1) ~= 0.76 of the econ half


def validate_panel(panel):
    if not isinstance(panel, dict) or not panel:
        raise ValueError('panel must be nonempty dict')
    for symbol, df in panel.items():
        if symbol not in SPECS:
            raise ValueError(f'unknown symbol: {symbol}')
        required = ['open', 'high', 'low', 'close', 'volume', 'contract', 'session', 'ct_sec']
        if not isinstance(df, pd.DataFrame) or not set(required) <= set(df.columns):
            raise ValueError(f'{symbol}: missing OHLCV/contract/session/ct_sec')
        if not isinstance(df.index, pd.DatetimeIndex) or str(df.index.tz) != 'UTC' or df.index.name != 'ts':
            raise ValueError(f'{symbol}: index must be ts in UTC')
        if not df.index.is_monotonic_increasing or not df.index.is_unique or df.empty:
            raise ValueError(f'{symbol}: timestamps must be nonempty sorted unique')
        if np.any(df.index.as_unit('ns').asi8 % MINUTE):
            raise ValueError(f'{symbol}: expected minute START timestamps')
        if df[required].isna().any().any():
            raise ValueError(f'{symbol}: null data')
        prices = df[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
        if not np.isfinite(prices).all() or not np.isfinite(df.volume.to_numpy(dtype=float)).all() or (df.volume < 0).any():
            raise ValueError(f'{symbol}: nonfinite OHLCV or negative volume')
        o, h, l, c = prices.T
        if np.any(h < np.maximum(o, c)) or np.any(l > np.minimum(o, c)) or np.any(h < l):
            raise ValueError(f'{symbol}: inconsistent OHLC')
        ct = df.index.tz_convert('America/Chicago')
        seconds = ct.hour * 3600 + ct.minute * 60 + ct.second
        if not np.array_equal(seconds, df.ct_sec.to_numpy()):
            raise ValueError(f'{symbol}: ct_sec disagrees with America/Chicago')
        if df.index[0] < pd.Timestamp(SPECS[symbol].available_from, tz='UTC'):
            raise ValueError(f'{symbol}: data predates actual product launch')
        if not df.contract.map(lambda x: isinstance(x, str) and bool(x)).all():
            raise ValueError(f'{symbol}: invalid contract')
        # Session labels must advance monotonically; no revisit after next session.
        session = df.session.astype(str).to_numpy()
        if np.any(session[1:] < session[:-1]):
            raise ValueError(f'{symbol}: nonmonotonic session labels')


def validate_targets(panel, targets):
    if not isinstance(targets, dict) or set(targets) - set(panel):
        raise ValueError('signals must return dict with only loaded symbols')
    result = {}
    for symbol, df in panel.items():
        values = np.asarray(targets.get(symbol, np.zeros(len(df), dtype=np.int64)))
        if values.shape != (len(df),) or values.dtype.kind not in 'iu':
            raise ValueError(f'{symbol}: targets must be same-length integer vector')
        if np.any(np.abs(values.astype(float)) * SPECS[symbol].weight > MAX_EXPOSURE + 1e-9):
            raise ValueError(f'{symbol}: target exceeds exposure limit')
        result[symbol] = values.astype(np.int64)
    return result


def load_panel(data, symbols):
    """Only explicit development paths. No discovery, resampling or micro proxying."""
    root = Path(data).expanduser().resolve()
    panel = {}
    for symbol in symbols:
        if symbol not in SPECS:
            raise ValueError(f'unknown symbol: {symbol}')
        # Caller supplies one explicit partition; sandbox controls accessible data.
        path = root / f'{symbol}.parquet'
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError('development paths cannot be symlinks')
        panel[symbol] = pd.read_parquet(path)
    validate_panel(panel)
    return panel


def lookahead_check(signals, panel, full=None):
    """Common UTC cutoffs: all markets truncated together, warmup retained."""
    if full is None:
        full = validate_targets(panel, signals({s: d.copy() for s, d in panel.items()}))
    timeline = pd.DatetimeIndex(sorted(set().union(*(d.index for d in panel.values()))))
    for fraction in (.25, .5, .75, .9):
        cut = timeline[min(len(timeline)-1, int(len(timeline) * fraction))]
        part = {s: d.loc[d.index <= cut].copy() for s, d in panel.items()}
        part = {s: d for s, d in part.items() if len(d)}
        got = validate_targets(part, signals(part))
        for symbol, df in part.items():
            if not np.array_equal(full[symbol][:len(df)], got[symbol]):
                raise ValueError(f'prefix causality failed: {symbol} at {cut}')
    return full


def validate_entry_risk(panel, risks):
    """Require causal nonnegative dollar risk per newly opened contract."""
    result = {}
    for symbol, df in panel.items():
        if symbol not in risks:
            raise ValueError(f'missing entry risk for {symbol}')
        values = np.asarray(risks[symbol])
        if values.ndim != 1 or len(values) != len(df) or not np.issubdtype(values.dtype, np.number):
            raise ValueError(f'entry risk shape/type invalid for {symbol}')
        values = values.astype(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f'entry risk must be finite and nonnegative for {symbol}')
        result[symbol] = values
    extra = set(risks) - set(panel)
    if extra:
        raise ValueError(f'entry risk unknown symbols: {sorted(extra)}')
    return result


def run_days(panel, targets, stressed=True, ledger=False, entry_risk=None):
    """One synchronized integer-position account, shared DLL and exposure cap.

    Target on row i close can fill only row i+1 open, exactly +1min, same
    contract/session. Batch orders validated together: never prioritize symbols.
    Bars ending at 15:10 flatten. No entry from 15:10 until 17:00 CT.
    """
    validate_panel(panel)
    targets = validate_targets(panel, targets)
    entry_risk = validate_entry_risk(panel, entry_risk or {symbol: np.zeros(len(df)) for symbol, df in panel.items()})
    symbols = sorted(panel)
    arrays = {}
    for symbol in symbols:
        df = panel[symbol]
        arrays[symbol] = {col: df[col].to_numpy() for col in ['open', 'high', 'low', 'close', 'ct_sec', 'contract']}
        arrays[symbol]['time'] = df.index.as_unit('ns').asi8
        arrays[symbol]['session'] = df.session.astype(str).to_numpy()
    # Each generator must bind its symbol, not capture the final loop variable.
    streams = [zip(a['time'], itertools.repeat(s), range(len(a['time']))) for s, a in arrays.items()]
    events = heapq.merge(*streams)
    positions = dict.fromkeys(symbols, 0)
    basis = dict.fromkeys(symbols, 0.)
    marks = dict.fromkeys(symbols, 0.)
    rows, fills = [], []
    per_market = {s: {'pnl': 0., 'fees': 0., 'contracts': 0, 'entries': 0, 'entries_0830_0930': 0, 'pnl_after_1000': 0.} for s in symbols}
    session, cash, day_min, locked, traded = None, 0., 0., False, False
    max_exposure = 0.
    max_entry_risk = 0.
    entry_risk_violations = []

    def fill(symbol, want, price, i, timestamp, reason):
        nonlocal cash, traded, max_entry_risk
        old = positions[symbol]
        delta = int(want - old)
        if not delta:
            return
        spec = SPECS[symbol]
        slip, fee = spec.costs(stressed)
        px = float(price + np.sign(delta) * slip)
        commission = abs(delta) * fee
        closed = min(abs(old), abs(delta)) if old * delta < 0 else 0
        realized = closed * np.sign(old) * (px - basis[symbol]) * spec.point_value
        if not old or old * want < 0:
            basis[symbol] = px
        elif old * delta > 0:
            basis[symbol] = (abs(old) * basis[symbol] + abs(delta) * px) / abs(want)
        cash -= delta * px * spec.point_value + commission
        pnl = realized - commission
        per_market[symbol]['pnl'] += pnl
        per_market[symbol]['fees'] += commission
        per_market[symbol]['contracts'] += abs(delta)
        sec = int(arrays[symbol]['ct_sec'][i])
        if 36000 <= sec < FLAT_SEC:
            per_market[symbol]['pnl_after_1000'] += pnl
        opened = max(0, abs(want) - abs(old)) if old * want >= 0 else abs(want)
        if opened:
            risk = opened * entry_risk[symbol][i - 1]
            max_entry_risk = max(max_entry_risk, risk)
            if risk > 300 + 1e-9:
                entry_risk_violations.append({'symbol': symbol, 'timestamp': pd.Timestamp(timestamp, tz='UTC').isoformat(),
                                              'risk': risk, 'opened_contracts': opened, 'risk_per_contract': entry_risk[symbol][i]})
            traded = True
            per_market[symbol]['entries'] += 1
            per_market[symbol]['entries_0830_0930'] += int(30600 <= sec < 34200)
        positions[symbol] = int(want)
        if ledger:
            fills.append({'timestamp': pd.Timestamp(timestamp, tz='UTC').isoformat(), 'session': session,
                          'symbol': symbol, 'contract': str(arrays[symbol]['contract'][i]),
                          'quantity': delta, 'price': px, 'commission': commission,
                          'realized_pnl': realized, 'position_before': old, 'position_after': int(want), 'reason': reason})

    def equity(prices):
        return cash + sum(positions[s] * prices[s] * SPECS[s].point_value for s in symbols)

    for timestamp, group in itertools.groupby(events, key=lambda x: x[0]):
        batch = [(s, i) for _, s, i in group]
        labels = {str(arrays[s]['session'][i]) for s, i in batch}
        if len(labels) != 1:
            raise ValueError('markets disagree on session at common timestamp')
        label = labels.pop()
        if session != label:
            if any(positions.values()):
                raise AssertionError('session transition must be flat')
            if session is not None:
                rows.append({'session': session, 'pnl': cash, 'minimum': day_min, 'traded': traded})
            session, cash, day_min, locked, traded = label, 0., 0., False, False
        # Orders batch at same open. Validate final exposure before any execution.
        wants = positions.copy()
        for symbol, i in batch:
            a = arrays[symbol]
            sec = int(a['ct_sec'][i])
            allowed = sec < FLAT_SEC or sec >= 61200
            valid = i > 0 and timestamp - int(a['time'][i-1]) == MINUTE and a['contract'][i] == a['contract'][i-1] and a['session'][i] == a['session'][i-1]
            wants[symbol] = int(targets[symbol][i-1]) if valid and allowed and not locked else 0
            marks[symbol] = float(a['open'][i])
        exposure = sum(abs(wants[s]) * SPECS[s].weight for s in symbols)
        if exposure > MAX_EXPOSURE + 1e-9:
            raise ValueError(f'aggregate requested exposure {exposure:g} > {MAX_EXPOSURE:g} at {pd.Timestamp(timestamp, tz="UTC")}')
        # Existing inventory can gap through DLL before new target orders.
        if equity(marks) <= -DLL:
            locked = True
            for symbol, i in batch:
                fill(symbol, 0, arrays[symbol]['open'][i], i, timestamp, 'dll_gap')
        if not locked:
            for symbol, i in batch:
                fill(symbol, wants[symbol], arrays[symbol]['open'][i], i, timestamp, 'signal')
        max_exposure = max(max_exposure, sum(abs(positions[s]) * SPECS[s].weight for s in symbols))
        adverse = marks.copy()
        for symbol, i in batch:
            a = arrays[symbol]
            adverse[symbol] = float(a['low'][i] if positions[symbol] > 0 else a['high'][i])
        worst = equity(adverse)
        day_min = min(day_min, worst)
        if worst <= -DLL and not locked:
            locked = True
            for symbol, i in batch:
                fill(symbol, 0, adverse[symbol], i, timestamp + MINUTE, 'dll_intrabar')
        for symbol, i in batch:
            a = arrays[symbol]
            marks[symbol] = float(a['close'][i])
            end = i + 1 == len(a['time'])
            discontinuity = not end and (int(a['time'][i+1]) - timestamp != MINUTE or a['contract'][i+1] != a['contract'][i] or a['session'][i+1] != a['session'][i])
            if end or discontinuity or int(a['ct_sec'][i]) == FLAT_SEC - 60 or FLAT_SEC <= int(a['ct_sec'][i]) < 61200:
                reason = 'data_end' if end else ('gap_or_roll' if discontinuity else 'daily_flat')
                fill(symbol, 0, a['close'][i], i, timestamp + MINUTE, reason)
        day_min = min(day_min, equity(marks))
    if session is not None:
        rows.append({'session': session, 'pnl': cash, 'minimum': day_min, 'traded': traded})
    entries = sum(x['entries'] for x in per_market.values())
    opening = sum(x['entries_0830_0930'] for x in per_market.values())
    share = opening / entries if entries else 0.
    families = sorted({SPECS[s].family for s in symbols if per_market[s]['entries']})
    return {'days': rows, 'ledger': fills, 'per_market': per_market, 'max_exposure': max_exposure,
            'max_entry_risk': max_entry_risk, 'entry_risk_violations': entry_risk_violations,
            'entry_risk_pass': not entry_risk_violations,
            'entry_0830_0930_share': share, 'entry_diversity_pass': share <= .4,
            'families_traded': families, 'family_count': len(families)}


def attempts(days):
    result = []
    for start in range(max(0, len(days) - WINDOW + 1)):
        balance, peak, best, trading_days = START, START, 0., 0
        passed, reason, duration = False, 'timeout', WINDOW
        for offset, day in enumerate(days[start:start + WINDOW]):
            floor = min(START, peak - MLL)
            if balance + day['minimum'] <= floor:
                reason, duration = 'mll', offset + 1
                break
            balance += day['pnl']
            trading_days += bool(day.get('traded', True))
            peak, best = max(peak, balance), max(best, day['pnl'])
            profit = balance - START
            if profit >= TARGET and trading_days >= 2 and best < .5 * profit:
                passed, reason, duration = True, 'passed', offset + 1
                break
        result.append({'start': days[start]['session'], 'passed': passed, 'days': duration, 'reason': reason})
    return result


def score(days):
    yearly = {}
    for year, group in itertools.groupby(days, key=lambda d: str(d['session'])[:4]):
        records = list(group)
        runs = attempts(records)
        yearly[year] = {'sessions': len(records), 'attempts': len(runs), 'passes': sum(r['passed'] for r in runs),
                        'rate': sum(r['passed'] for r in runs) / len(runs) if runs else None,
                        'pnl': float(sum(r['pnl'] for r in records))}
    rates = sorted(x['rate'] for x in yearly.values() if x['rate'] is not None)
    worst = float(np.mean(rates[:3])) if rates else 0.
    recent = [x['rate'] for y, x in yearly.items() if '2020' <= y <= '2023' and x['rate'] is not None]
    recent_pass = all(r > 0 for r in recent) if recent else None
    folds = fold_rates(days)
    ranked = sorted(folds.values())
    fold_score = float(np.mean(ranked[:DEV_WORST_FOLDS])) if ranked else 0.
    consistency = (sum(1 for x in yearly.values() if x['pnl'] > 0) / len(yearly)) if yearly else 0.
    total_pnl = float(sum(x['pnl'] for x in yearly.values()))
    # Economic gradient. Pass rate alone stays 0 until a strategy clears the combine
    # target, so a purely pass-based dev_score would repeat the v3.2 failure in softer
    # form: economically improving candidates all read 0 and agents get no signal.
    # econ_grade is bounded [0,1] and monotone in both year consistency and
    # dollars-per-year, so improvement is visible before the pass threshold is crossed.
    per_year = total_pnl / len(yearly) if yearly else 0.
    econ_grade = float(np.clip(.5 * consistency + .5 * np.tanh(per_year / ECON_PNL_SCALE), 0., 1.))
    dev_raw = DEV_PASS_WEIGHT * fold_score * consistency + (1. - DEV_PASS_WEIGHT) * econ_grade
    return {'score': worst if recent_pass is not False else 0., 'worst_three_mean': worst,
            'yearly': yearly, 'recent_2020_2023_gate': recent_pass,
            'folds': folds, 'fold_score': fold_score, 'positive_year_fraction': consistency,
            'total_pnl': total_pnl, 'pnl_per_year': per_year, 'econ_grade': econ_grade,
            'dev_score_raw': dev_raw,
            'attempts': attempts(days),
            'score_definition': ('final: mean worst min(3,eligible_years) yearly rates, recent 2020-2023 gate. '
                                 'dev_score_raw: %g * (mean worst %d of %d contiguous fold pass rates * positive-year '
                                 'fraction) + %g * econ_grade; caller subtracts knob penalty'
                                 % (DEV_PASS_WEIGHT, DEV_WORST_FOLDS, N_FOLDS, 1. - DEV_PASS_WEIGHT))}


def fold_rates(days):
    """Pass rate per contiguous era fold, equal-count session blocks in chronological
    order. A candidate cannot win by fitting one era."""
    sessions = sorted({str(d['session']) for d in days})
    if not sessions:
        return {}
    out = {}
    for i, chunk in enumerate(np.array_split(np.array(sessions), min(N_FOLDS, len(sessions)))):
        members = set(chunk.tolist())
        runs = attempts([d for d in days if str(d['session']) in members])
        out['fold%d' % i] = (sum(r['passed'] for r in runs) / len(runs)) if runs else 0.
    return out


def count_knobs(path):
    """Free-parameter count for the BIC-style dev penalty. Prefer an explicit
    PARAMS mapping in strategy.py; otherwise fall back to counting distinct numeric
    literals in the AST (0/1/-1 excluded). Declaring PARAMS is cheaper and explicit,
    so the fallback is deliberately blunt."""
    src = Path(path).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == 'PARAMS' for t in node.targets):
            if isinstance(node.value, (ast.Dict, ast.List, ast.Tuple)):
                return len(node.value.keys if isinstance(node.value, ast.Dict) else node.value.elts), 'declared PARAMS'
    lits = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))
            and not isinstance(n.value, bool) and n.value not in (0, 1, -1, 0., 1., -1.)}
    return len(lits), 'ast numeric literals (declare PARAMS to make this explicit)'


def apply_knob_penalty(dev_score_raw, knobs):
    return max(0., dev_score_raw - KNOB_PENALTY * knobs)



def permute_targets(panel, targets, seed):
    """Honest side null: one random sign per SESSION shared by all markets.

    Preserves exact integer sizes, entry times, holding/flat pattern and aggregate
    exposure; does not randomize schedule. Reversal signs flip together per session.
    This tests side choice only, not timing edge. 200 seeds intended finalists only.
    """
    rng = np.random.default_rng(seed)
    sessions = sorted(set().union(*(set(df.session.astype(str)) for df in panel.values())))
    signs = dict(zip(sessions, rng.choice([-1, 1], len(sessions))))
    return {s: values * np.array([signs[x] for x in panel[s].session.astype(str)], dtype=np.int64) for s, values in targets.items()}


def economic_eligibility(days, diversity_pass, entry_risk_pass, null=None, mode='dev'):
    yearly_pnl = {year: sum(day['pnl'] for day in group)
                  for year, group in itertools.groupby(days, key=lambda day: str(day['session'])[:4])}
    total_pnl = sum(yearly_pnl.values())
    positive_years = sum(pnl > 0 for pnl in yearly_pnl.values())
    n_years = len(yearly_pnl)
    year_frac = positive_years / n_years if n_years else 0.
    null_p = None if null is None else null['p_value']
    # mode='dev' is the graded exploration gate: year robustness is a floor, not a
    # cliff, and the null check binds only when a null was actually run. mode='final'
    # is the sealed gate and keeps the strict v3.2 conjunctive rules.
    if mode == 'final':
        year_ok = positive_years == n_years
        null_ok = null_p is not None and null_p <= .05 and null.get('seeds') == 200
        year_key = 'all_partition_years_positive'
    else:
        year_ok = year_frac >= DEV_YEAR_FLOOR
        null_ok = True if null_p is None else null_p <= .05
        year_key = 'year_fraction_at_least_%g' % DEV_YEAR_FLOOR
    checks = {'positive_stressed_pnl': total_pnl > 0, year_key: year_ok,
              'entry_diversity': diversity_pass,
              'entry_risk_at_most_300': entry_risk_pass,
              'side_null_p_le_0_05': null_ok}
    return {'passed': all(checks.values()), 'mode': mode, 'checks': checks, 'total_pnl': total_pnl,
            'yearly_pnl': yearly_pnl, 'positive_years': positive_years,
            'partition_years': n_years, 'positive_year_fraction': year_frac, 'null_p_value': null_p}


def entry_risks(module, panel):
    risks = getattr(module, 'entry_risk', None)
    if risks is None:
        return {symbol: np.full(len(df), 301.) for symbol, df in panel.items()}
    return validate_entry_risk(panel, risks({symbol: df.copy() for symbol, df in panel.items()}))


def null_benchmark(panel, targets, observed, risks, seeds=0):
    if seeds not in NULL_SEEDS_ALLOWED:
        raise ValueError('--null-seeds must be one of %s: 0 screen, 20 provisional dev, 200 frozen finalist'
                         % (NULL_SEEDS_ALLOWED,))
    if not seeds:
        return None
    # Compare the ungated statistic. v3.2 compared the eligibility-zeroed 'score',
    # so any candidate failing the recent-year gate scored 0 against nulls that also
    # scored 0 and got p=1.0 by construction. worst_three_mean is the honest pass rate.
    scores = [score(run_days(panel, permute_targets(panel, targets, seed),
                             entry_risk=risks)['days'])['worst_three_mean'] for seed in range(seeds)]
    return {'kind': 'session_shared_side_permutation', 'seeds': seeds, 'scores': scores,
            'statistic': 'worst_three_mean',
            'provisional': seeds < 200,
            'p_value': (1 + sum(x >= observed for x in scores)) / (seeds + 1),
            'note': ('shared session direction only; preserves entries, size, and timing. '
                     'PROVISIONAL wide CI, dev ranking only' if seeds < 200 else
                     'shared session direction only; preserves entries, size, and timing')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--symbols', nargs='+', default=['MES', 'MGC'])
    parser.add_argument('--strategy', default=str(Path(__file__).resolve().parents[1] / 'strategy.py'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--ledger', help='write reference stressed fills JSONL')
    parser.add_argument('--null-seeds', type=int, default=0,
                        help='0 fast screen; 20 provisional dev directional check; 200 frozen finalist only')
    parser.add_argument('--mode', choices=('dev', 'final'), default='dev',
                        help='dev = graded exploration gates; final = strict sealed gates')
    args = parser.parse_args()
    panel = load_panel(args.data, args.symbols)
    spec = importlib.util.spec_from_file_location('candidate_strategy', args.strategy)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    targets = validate_targets(panel, module.signals({s: df.copy() for s, df in panel.items()}))
    risks = entry_risks(module, panel)
    lookahead_check(module.signals, panel, targets)
    stressed = run_days(panel, targets, stressed=True, ledger=bool(args.ledger), entry_risk=risks)
    base = run_days(panel, targets, stressed=False, entry_risk=risks)
    report = {'schema_version': 3, 'strategy_sha256': hashlib.sha256(Path(args.strategy).read_bytes()).hexdigest(),
              'symbols': sorted(panel), 'stressed': {**stressed, **score(stressed['days'])},
              'base': {**base, **score(base['days'])}, 'null': None,
              'costs': {s: {'tick': SPECS[s].tick, 'point_value': SPECS[s].point_value,
                            'weight': SPECS[s].weight, 'base': SPECS[s].costs(False), 'stressed': SPECS[s].costs(True)} for s in panel},
              'limitations': ['OHLC aggregate extremes assume simultaneous adverse moves', 'gap/roll flatten at prior close',
                              'reference ledger, rolling MLL attempts stop separately', 'candidate process requires external sandbox',
                              'fees and exposure weights are estimates; no live broker guarantee', 'no null pass inferred unless human protocol assesses diagnostics']}
    report['stressed']['score_before_eligibility'] = report['stressed']['score']
    knobs, knob_method = count_knobs(args.strategy)
    report['stressed']['knobs'] = knobs
    report['stressed']['knob_method'] = knob_method
    report['stressed']['knob_penalty'] = KNOB_PENALTY * knobs
    # dev_score is the graded leaderboard number: never forced to zero, so agents get
    # a gradient from experiment one. 'score' stays the strict eligibility-gated value.
    report['stressed']['dev_score'] = apply_knob_penalty(report['stressed']['dev_score_raw'], knobs)
    report['base']['dev_score'] = apply_knob_penalty(report['base']['dev_score_raw'], knobs)
    report['null'] = null_benchmark(panel, targets, report['stressed']['worst_three_mean'], risks, args.null_seeds)
    eligibility = economic_eligibility(stressed['days'], stressed['entry_diversity_pass'],
                                       stressed['entry_risk_pass'], report['null'], args.mode)
    if args.mode == 'final' and report['null'] is None:
        eligibility['checks']['side_null_p_le_0_05'] = False
        eligibility['passed'] = False
    report['stressed']['eligibility'] = eligibility
    if not eligibility['passed']:
        report['stressed']['score'] = 0.
    if args.ledger:
        with open(args.ledger, 'w') as fh:
            for row in stressed['ledger']:
                fh.write(json.dumps(row, allow_nan=False) + '\n')
    report['stressed'].pop('ledger')
    report['base'].pop('ledger')
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    s = report['stressed']
    print(f"dev_score: {s['dev_score']:.6f}   (pass {s['fold_score']:.6f} * years {s['positive_year_fraction']:.3f} "
          f"weighted {DEV_PASS_WEIGHT:g}, econ_grade {s['econ_grade']:.4f} weighted {1 - DEV_PASS_WEIGHT:g}, "
          f"- knob_penalty {s['knob_penalty']:.6f} for {knobs} knobs via {knob_method})")
    print(f"folds: {json.dumps(s['folds'])}")
    print(f"score (eligibility-gated, mode={args.mode}): {s['score']:.6f}   "
          f"worst_three_mean {s['worst_three_mean']:.6f}   eligible={eligibility['passed']}")
    if report['null']:
        print(f"null: seeds={report['null']['seeds']} p={report['null']['p_value']:.4f} "
              f"provisional={report['null']['provisional']}")
    print(f"eligibility checks: {json.dumps(eligibility['checks'])}")
    for year, stats in s['yearly'].items():
        print(f"year {year}: {json.dumps(stats)}")
    print(json.dumps({'dev_score': s['dev_score'], 'score': s['score'],
                      'eligible': eligibility['passed'], 'output': args.output}))


if __name__ == '__main__':
    main()
