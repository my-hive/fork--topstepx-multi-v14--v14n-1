"""Runnable without pytest: python eval/test_sim.py."""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sim import (run_days, attempts, score, validate_panel, validate_targets,
                 lookahead_check, permute_targets, load_panel, main, economic_eligibility,
                 validate_entry_risk)
from instruments import SPECS


def frame(times=None, prices=None, symbol='MES'):
    index = pd.DatetimeIndex(times if times is not None else pd.date_range('2024-01-02 16:00', periods=4, freq='min', tz='UTC'), name='ts')
    values = np.asarray(prices if prices is not None else [100., 100., 105., 110.])
    ct = index.tz_convert('America/Chicago')
    return pd.DataFrame({'open': values, 'high': values + 1, 'low': values - 1, 'close': values,
                         'volume': 10, 'contract': symbol + 'H4', 'session': '2024-01-02',
                         'ct_sec': ct.hour * 3600 + ct.minute * 60}, index=index)


def rejects(fn):
    try:
        fn()
    except (ValueError, AssertionError):
        return
    raise AssertionError('expected rejection')


def check():
    df = frame()
    target = {'MES': np.array([1, 1, 0, 0])}
    r = run_days({'MES': df}, target, stressed=False, ledger=True)
    fills = r['ledger']
    assert len(fills) == 2 and fills[0]['timestamp'] == df.index[1].isoformat()
    assert fills[0]['position_after'] == 1 and fills[1]['timestamp'] == df.index[3].isoformat()
    assert fills[0]['price'] == 100.25 and fills[1]['price'] == 109.75
    assert abs(r['days'][0]['pnl'] - (9.5 * 5 - 1.5)) < 1e-9
    assert abs(sum(f['realized_pnl'] - f['commission'] for f in fills) - r['days'][0]['pnl']) < 1e-9
    assert run_days({'MES': df}, target)['days'][0]['pnl'] < r['days'][0]['pnl']
    # No same bar, last-bar target never fills.
    r = run_days({'MES': df}, {'MES': np.array([0, 0, 0, 1])}, ledger=True)
    assert not r['ledger']
    # Same account, integer holdings, simultaneous exposure rejected not prioritized.
    panel = {'MES': df, 'MGC': frame(symbol='MGC')}
    rejects(lambda: run_days(panel, {s: np.full(4, 30) for s in panel}))
    r = run_days(panel, {s: np.full(4, 20) for s in panel}, ledger=True)
    assert r['max_exposure'] == 4 and all(isinstance(x['position_after'], int) for x in r['ledger'])
    assert abs(sum(r['per_market'][s]['pnl'] for s in panel) - r['days'][0]['pnl']) < 1e-8
    # Gap/roll closes at prior close; no pending carry to disconnected open.
    gap = frame(times=pd.to_datetime(['2024-01-02 16:00Z', '2024-01-02 16:01Z', '2024-01-02 16:10Z', '2024-01-02 16:11Z']))
    r = run_days({'MES': gap}, {'MES': np.ones(4, dtype=int)}, ledger=True)
    assert r['ledger'][1]['reason'] == 'gap_or_roll'
    assert r['ledger'][1]['timestamp'] == '2024-01-02T16:02:00+00:00'
    assert not any(x['timestamp'] == gap.index[2].isoformat() for x in r['ledger'])
    roll = df.copy()
    roll.loc[roll.index[2]:, 'contract'] = 'MESM4'
    r = run_days({'MES': roll}, {'MES': np.ones(4, dtype=int)}, ledger=True)
    assert r['ledger'][1]['contract'] == 'MESH4' and r['ledger'][1]['reason'] == 'gap_or_roll'
    # DLL gap overshoots; no magical -1000 fill, shared lock forbids later entries.
    crash = frame(prices=[100., 100., 50., 50.])
    r = run_days({'MES': crash}, {'MES': np.full(4, 10)}, ledger=True)
    assert r['days'][0]['pnl'] < -2500 and r['ledger'][-1]['reason'] == 'dll_gap'
    assert len(r['ledger']) == 2
    # Intrabar multiasset DLL uses sum adverse extremes, not independent asset limits.
    d1, d2 = frame(), frame(symbol='MGC')
    d1.loc[d1.index[1], 'low'] = 80
    d2.loc[d2.index[1], 'low'] = 90
    r = run_days({'MES': d1, 'MGC': d2}, {'MES': np.full(4, 5), 'MGC': np.full(4, 5)}, ledger=True)
    assert r['days'][0]['pnl'] < -1000 and sum(f['reason'] == 'dll_intrabar' for f in r['ledger']) == 2
    # 15:10 flat, no opening until 17:00. Target close at 17:00 fills 17:01.
    evening = frame(times=pd.to_datetime(['2024-01-02 21:08Z', '2024-01-02 21:09Z', '2024-01-02 21:10Z', '2024-01-02 21:11Z']))
    r = run_days({'MES': evening}, {'MES': np.ones(4, dtype=int)}, ledger=True)
    assert r['ledger'][1]['timestamp'] == '2024-01-02T21:10:00+00:00'
    assert len(r['ledger']) == 2
    overnight = frame(times=pd.date_range('2024-01-02 22:59Z', periods=4, freq='min'))
    r = run_days({'MES': overnight}, {'MES': np.ones(4, dtype=int)}, ledger=True)
    assert r['ledger'][0]['timestamp'] == '2024-01-02T23:00:00+00:00'
    # Timestamp source CT timezone, including DST, not UTC hour or process TZ.
    summer = frame(times=pd.date_range('2024-07-02 20:08Z', periods=4, freq='min'))
    summer['session'] = '2024-07-02'
    r = run_days({'MES': summer}, {'MES': np.ones(4, dtype=int)}, ledger=True)
    assert r['ledger'][1]['timestamp'] == '2024-07-02T20:10:00+00:00'
    broken = df.copy(); broken['ct_sec'] = 0
    rejects(lambda: validate_panel({'MES': broken}))
    rejects(lambda: validate_targets({'MES': df}, {'XYZ': np.zeros(4, int)}))
    for values in ([0., 0., np.nan, 0.], [0., 0., 0., 0.], [1, 2], [[1], [2], [3], [4]]):
        rejects(lambda: validate_targets({'MES': df}, {'MES': values}))
    old = frame(times=pd.date_range('2010-01-02 16:00Z', periods=4, freq='min'))
    rejects(lambda: validate_panel({'MES': old}))
    assert SPECS['QM'].weight == .5 and SPECS['QG'].weight == .25 and SPECS['SIL'].weight == .2
    # strict consistency, minimum trading days and EOD trailing floor.
    days = [{'session': f'2024-01-{i+1:02}', 'pnl': 1000., 'minimum': 0., 'traded': True} for i in range(30)]
    assert attempts(days)[0]['passed'] and attempts(days)[0]['days'] == 3
    days[0]['pnl'] = 3000.
    for d in days[1:]: d['pnl'] = 0.
    assert not attempts(days)[0]['passed']
    days[1]['minimum'] = -3001.
    assert attempts(days)[0]['reason'] == 'mll'  # floor locked 50K after first +3K
    assert score(days)['yearly']['2024']['attempts'] == 1
    # Prefix causality, common cutoffs across uneven sparse markets.
    spec = importlib.util.spec_from_file_location('baseline', Path(__file__).parents[1] / 'strategy.py')
    baseline = importlib.util.module_from_spec(spec); spec.loader.exec_module(baseline)
    lookahead_check(baseline.signals, panel)
    rejects(lambda: lookahead_check(lambda p: {s: np.full(len(d), len(d), dtype=int) for s, d in p.items()}, panel))
    permuted = permute_targets(panel, {s: np.array([1, 1, -2, 0]) for s in panel}, 7)
    assert np.array_equal(abs(permuted['MES']), [1, 1, 2, 0])
    assert np.array_equal(permuted['MES'], permuted['MGC'])
    eligible = economic_eligibility([{'session': '2020-01-01', 'pnl': 1},
                                    {'session': '2021-01-01', 'pnl': 1},
                                    {'session': '2022-01-01', 'pnl': 1}], True, True,
                                   {'p_value': .01})
    assert eligible['passed']
    assert economic_eligibility([{'session': '2020-01-01', 'pnl': 1}], True, True,
                                {'p_value': .01})['passed']
    assert economic_eligibility([{'session': '2024-01-01', 'pnl': 1},
                                 {'session': '2025-01-01', 'pnl': 1}], True, True,
                                {'p_value': .01})['passed']
    assert not economic_eligibility([{'session': '2024-01-01', 'pnl': 1},
                                     {'session': '2025-01-01', 'pnl': -1}], True, True,
                                    {'p_value': .01})['passed']
    assert validate_entry_risk({'MES': df}, {'MES': np.zeros(len(df))})['MES'].sum() == 0
    rejects(lambda: validate_entry_risk({'MES': df}, {'MES': np.full(len(df), -1.)}))
    risky = run_days({'MES': df}, {'MES': np.array([0, 2, 0, 0])},
                     entry_risk={'MES': np.full(len(df), 151.)})
    assert not risky['entry_risk_pass'] and risky['max_entry_risk'] == 302
    # CLI and machine report/ledger integration without modifying repository root.
    import subprocess, sys
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for s, d in panel.items():
            d.to_parquet(root/f'{s}.parquet')
        loaded = load_panel(root, list(panel))
        assert len(loaded) == 2
        strategy = root/'candidate.py'
        strategy.write_text('import numpy as np\ndef signals(panel):\n return {s: (np.arange(len(d)) < 2).astype(int) for s,d in panel.items()}\n')
        subprocess.run([sys.executable, str(Path(__file__).with_name('sim.py')), '--data', temp, '--symbols', 'MES', 'MGC', '--strategy', str(strategy), '--output', str(root/'report.json'), '--ledger', str(root/'fills.jsonl'), '--null-seeds', '200'], check=True)
        import json
        report = json.loads((root/'report.json').read_text())
        assert report['schema_version'] == 3 and len((root/'fills.jsonl').read_text().splitlines()) == 4
        assert 'eligibility' in report['stressed']
        assert report['null']['seeds'] == 200
        # v1.4: graded dev score is always present and never gated to zero by eligibility.
        s = report['stressed']
        assert 'dev_score' in s and 'dev_score_raw' in s and 'folds' in s
        assert s['dev_score'] == max(0., s['dev_score_raw'] - s['knob_penalty'])
        assert s['knobs'] >= 0 and s['knob_penalty'] == 0.001 * s['knobs']
        assert report['null']['statistic'] == 'worst_three_mean'
        assert report['stressed']['eligibility']['mode'] == 'dev'
    # v1.4 pure-function checks: knob counting, penalty, fold split, dev/final gates.
    from sim import count_knobs, apply_knob_penalty, fold_rates, null_benchmark, NULL_SEEDS_ALLOWED
    with tempfile.TemporaryDirectory() as temp:
        declared = Path(temp)/'declared.py'
        declared.write_text('PARAMS = {"n": 20, "k": 3.5, "z": 7}\nimport numpy as np\n'
                            'def signals(panel):\n    return {s: np.zeros(len(d), int) for s, d in panel.items()}\n')
        assert count_knobs(declared)[0] == 3 and count_knobs(declared)[1] == 'declared PARAMS'
        implicit = Path(temp)/'implicit.py'
        implicit.write_text('import numpy as np\nA = 20\nB = 3.5\n'
                            'def signals(panel):\n    return {s: np.zeros(len(d), int) for s, d in panel.items()}\n')
        assert count_knobs(implicit)[0] == 2 and 'ast' in count_knobs(implicit)[1]
    assert apply_knob_penalty(0.01, 4) == 0.01 - 0.004
    assert apply_knob_penalty(0.001, 5) == 0.  # penalty floors at zero, never negative
    # fold_rates splits chronologically into N_FOLDS contiguous blocks
    many = [{'session': '20%02d-01-01' % (10 + i // 5), 'pnl': 100., 'minimum': -50., 'traded': True}
            for i in range(40)]
    fr = fold_rates(many)
    assert len(fr) == 4 and all(v == 0. for v in fr.values())  # no attempt can pass on 5-session folds
    # dev gate is a floor, final gate is a cliff
    days = [{'session': '%d-01-01' % (2010 + i), 'pnl': 100. if i % 4 else -50., 'adverse': 0., 'fills': []}
            for i in range(10)]
    dev = economic_eligibility(days, True, True, None, 'dev')
    fin = economic_eligibility(days, True, True, {'p_value': .01, 'seeds': 200}, 'final')
    assert dev['checks']['year_fraction_at_least_0.7'] and dev['passed']
    assert not fin['checks']['all_partition_years_positive'] and not fin['passed']
    # final mode rejects a 20-seed provisional null even at p<=0.05
    prov = economic_eligibility(days, True, True, {'p_value': .01, 'seeds': 20}, 'final')
    assert not prov['checks']['side_null_p_le_0_05']
    # dev mode tolerates an unrun null but binds when one exists
    assert economic_eligibility(days, True, True, None, 'dev')['checks']['side_null_p_le_0_05']
    assert not economic_eligibility(days, True, True, {'p_value': .40, 'seeds': 20}, 'dev')['checks']['side_null_p_le_0_05']
    assert NULL_SEEDS_ALLOWED == (0, 20, 200)
    # v1.4 core property: economic improvement is visible BEFORE the pass threshold.
    # A candidate with positive PnL in most years but zero combine passes must still
    # earn a nonzero dev_score_raw, otherwise the leaderboard has no gradient again.
    improving = [{'session': '%d-01-0%d' % (2010 + i // 5, 1 + i % 5), 'pnl': 120.,
                  'minimum': -50., 'traded': True} for i in range(40)]
    losing = [{'session': '%d-01-0%d' % (2010 + i // 5, 1 + i % 5), 'pnl': -120.,
               'minimum': -50., 'traded': True} for i in range(40)]
    si, sl = score(improving), score(losing)
    assert all(v == 0. for v in si['folds'].values())  # no combine pass either way
    assert si['dev_score_raw'] > 0., 'positive economics must produce a nonzero gradient'
    assert sl['dev_score_raw'] == 0., 'uniformly negative economics must floor at zero'
    assert si['econ_grade'] > sl['econ_grade']
    assert 0. <= si['econ_grade'] <= 1. and 0. <= sl['econ_grade'] <= 1.
    # more dollars per year => strictly higher econ_grade, bounded at 1
    richer = score([{**d, 'pnl': 5000.} for d in improving])
    assert richer['econ_grade'] > si['econ_grade'] and richer['econ_grade'] <= 1.
    # knob penalty can push a small but real dev_score back to zero, never below
    assert apply_knob_penalty(si['dev_score_raw'], 10 ** 6) == 0.
    for bad in (1, 19, 50, 199, 201):
        try:
            null_benchmark(None, None, 0., None, bad)
            raise AssertionError('seeds=%d should be rejected' % bad)
        except ValueError:
            pass
    assert null_benchmark(None, None, 0., None, 0) is None
    reversal = run_days({'MES': frame(prices=[100., 100., 105., 110.])},
                        {'MES': np.array([1, -1, 0, 0])}, stressed=False, ledger=True)
    ledger_net = sum(f['realized_pnl'] - f['commission'] for f in reversal['ledger'])
    assert abs(reversal['days'][0]['pnl'] - ledger_net) < 1e-8
    assert abs(ledger_net - (-8.0)) < 1e-8
    print('multiasset simulator self-check ok')


if __name__ == '__main__':
    check()
