"""v14n-1: MNQ afternoon intraday momentum.

Mechanism: morning (08:30->entry) RTH move direction persists into the close as
late-day hedging/rebalancing flows chase the day's trend (Gao-Han-Li-Zhou style
intraday momentum). One entry per session at PARAMS['entry_sec'] CT in the sign of
the morning move, protective stop = k * trailing mean of prior sessions'
entry->close absolute move, size = floor(risk cap / stop dollars). Flat by
evaluator 15:10 close. Gates: (1) entry time, (2) nonzero morning move,
(3) stop warmup available, (4) size >= 1.
"""
import numpy as np

PARAMS = {"entry_sec": 50400, "form_sec": 30600, "stop_k": 1.0, "lookback": 20}
SYMBOL, POINT_VALUE, RISK_CAP, MAX_QTY, LAST_SEC = 'MNQ', 2.0, 300.0, 50, 54480


def _plan(df):
    """Per row: target position and declared dollar risk per contract."""
    n = len(df)
    target = np.zeros(n, dtype=np.int64)
    risk = np.zeros(n)
    o, c = df.open.to_numpy(), df.close.to_numpy()
    sec = df.ct_sec.to_numpy()
    hist = []
    for idx in df.groupby('session', sort=False).indices.values():
        s = sec[idx]
        a = idx[s == PARAMS['form_sec']]
        b = idx[s == PARAMS['entry_sec']]
        e = idx[s == LAST_SEC]
        if len(hist) >= PARAMS['lookback'] and len(a) and len(b):
            stop = PARAMS['stop_k'] * np.mean(hist[-PARAMS['lookback']:])
            qty = min(MAX_QTY, int(RISK_CAP // (stop * POINT_VALUE))) if stop > 0 else 0
            # Morning move measured to close of bar b-1, completed before bar b opens.
            sig = int(np.sign(c[b[0] - 1] - o[a[0]]))
            if qty and sig:
                risk[idx] = stop * POINT_VALUE
                level = c[b[0] - 1] - sig * stop
                for j in idx[(s >= PARAMS['entry_sec'] - 60) & (s < LAST_SEC + 60)]:
                    if sig * (c[j] - level) <= 0:
                        break
                    target[j] = sig * qty
        # history update only after the session is complete (causal for next session)
        if len(b) and len(e):
            hist.append(abs(c[e[0]] - c[b[0] - 1]))
    return target, risk


def _ok(panel):
    return SYMBOL in panel


def signals(panel):
    out = {s: np.zeros(len(df), dtype=np.int64) for s, df in panel.items()}
    if _ok(panel):
        out[SYMBOL] = _plan(panel[SYMBOL])[0]
    return out


def entry_risk(panel):
    out = {s: np.zeros(len(df)) for s, df in panel.items()}
    if _ok(panel):
        out[SYMBOL] = _plan(panel[SYMBOL])[1]
    return out
