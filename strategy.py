"""v14n-1: MNQ afternoon intraday momentum.

Mechanism: morning (08:30->entry) RTH move direction persists into the close as
late-day hedging/rebalancing flows chase the day's trend (Gao-Han-Li-Zhou style
intraday momentum). One entry per session at PARAMS['entry_sec'] CT in the sign of
the morning move, protective stop = k * trailing mean of prior sessions'
entry->close absolute move, size = floor(risk cap / stop dollars). Exit 15:00 CT
before closing-imbalance reversal. Gates: (1) entry time, (2) nonzero morning move,
(3) stop warmup available, (4) size >= 1,
(5) one add-on at add_sec only if open profit (pyramid on confirmed trend).
"""
import numpy as np

PARAMS = {"entry_sec": 50400, "form_sec": 30600, "stop_k": 1.0, "lookback": 20, "exit_sec": 54000, "add_sec": 52200}
SYMBOL, POINT_VALUE, RISK_CAP, MAX_QTY = 'MNQ', 2.0, 300.0, 50


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
        e = idx[s == PARAMS['exit_sec'] - 60]
        if len(hist) >= PARAMS['lookback'] and len(a) and len(b):
            stop = PARAMS['stop_k'] * np.mean(hist[-PARAMS['lookback']:])
            qty = min(MAX_QTY, int(RISK_CAP // (stop * POINT_VALUE))) if stop > 0 else 0
            # Morning move measured to close of bar b-1, completed before bar b opens.
            sig = int(np.sign(c[b[0] - 1] - o[a[0]]))
            if qty and sig:
                risk[idx] = stop * POINT_VALUE
                level = c[b[0] - 1] - sig * stop
                pos = qty
                for j in idx[(s >= PARAMS['entry_sec'] - 60) & (s < PARAMS['exit_sec'] - 60)]:
                    gap = sig * (c[j] - level)
                    if gap <= 0:
                        break
                    # Pyramid once at add_sec if trade is in profit: added contracts
                    # declare risk to the same shared stop level (causal, prior bar).
                    if sec[j] == PARAMS['add_sec'] - 60 and gap > stop:
                        risk[j] = gap * POINT_VALUE
                        pos = min(MAX_QTY, qty + int(RISK_CAP // risk[j]))
                    target[j] = sig * pos
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
