# v14n-1 mechanism notes

Best: e13, source SHA256 in experiments.jsonl row 13. dev_score 0.2997, folds 0/.132/.287/.210,
positive_year_fraction 0.8, 7 knobs, ~$1.9k/yr stressed. MNQ only.

Mechanism: intraday momentum. Sign of 08:30 open -> 13:59 close on MNQ; enter 14:00 CT,
stop = 1.0 x trailing-20-session mean |14:00->15:00| move (declared as entry risk), qty = floor($300/stop$),
one add-on at 14:30 if open profit > stop (sized to same stop level), flat 15:00 (15:00-15:10 reverts).
Cost-room gate: skip when trailing PM move < 8 x stressed round trip.

Gates (5): entry time, nonzero morning move, warmup, qty>=1, cost room; add-on condition.

Evidence: 20-seed side null p=0.381 (e10), p=0.095 (e1). Direction evidence weak -> fails dev null gate.
2019 partial year negative in every variant. No transfer: MES/MYM/M2K/MGC/MCL lose; NQ 2010-2018 loses.
Rejected: stop 0.5/0.75/1.5, lookback 60, size gate on morning move, full-session formation, VWAP signal.

Sealed validation: none submitted. No candidate has eligible dev evidence (null p >> 0.05);
submitting would spend owner compute on a known-failing candidate.
