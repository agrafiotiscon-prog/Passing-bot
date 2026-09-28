"""Final configuration selection (train years only) and out-of-sample report.

Strategy ("bold play"): one market entry per day at a fixed server time,
direction from a trend filter (sign of the N-day return; N=0 -> coin flip),
stop = k * daily ATR(14), take-profit set so that a win lifts equity exactly
to the +10% target, risk per trade = min(2.9%, room to the daily / overall
limits). Flat at 23:45 server. Daily equity stop at -2.9%.
"""
import itertools
import sys

import numpy as np
import pandas as pd

import harness as h
import sim
import strategies as S

CFG = {
    "XAUUSD": dict(train=("2012-06-01", "2020-01-01"), test=("2020-01-01", None), weekdays=True, lev=50.0,
                   hours=(3.0, 10.0, 16.5)),
    "BTCUSD": dict(train=("2016-03-01", "2022-01-01"), test=("2022-01-01", None), weekdays=False, lev=3.3,
                   hours=(1.0, 10.0, 16.5)),
}


def trend_sig(df: pd.DataFrame, n: int, seed: int = 0):
    if n == 0:
        return f"random:{seed}"
    d = df.resample("1D").agg({"open": "first", "close": "last"}).dropna()
    return np.sign(np.log(d.close / d.close.shift(n))).shift(1)


def evaluate(g, sym, hour, n, sl, risk=0.029, halt=0.029, start=None, end=None, weekdays=True, lev=50.0,
             seeds=6, tf_df=None):
    df = g.bars[sym] if tf_df is None else tf_df
    outs = []
    for sd in (range(seeds) if n == 0 else [0]):
        tr = h.to_grid(g, sym, S.daily_fixed_time(df, hour, trend_sig(df, n, sd), sl, 3.5, exit_h=23.75))
        s, e = g.starts(start, end, 14, weekdays_only=weekdays)
        o = sim.run_all(s, e, g.O, g.H, g.L, g.C, g.day, g.HC, g.SL, tr.tk, tr.t0, tr.dirn, tr.stop, tr.tgt,
                        tr.t1, tr.ent, sim.params(risk=risk, halt=halt, maxpos=1, lev=lev, bold=1, bold_rrmax=20))
        r = pd.DataFrame(o, columns=["status", "end_bar", "final", "ntr", "mineq"], index=g.idx[s])
        r["days"] = (g.idx[r.end_bar.astype(int)] - r.index).total_seconds() / 86400
        r["seed"] = sd
        outs.append(r)
    return pd.concat(outs)


def by_year(r):
    f = lambda x: pd.Series({"pass": (x.status == 1).mean(), "fail": (x.status == -1).mean(),
                             "timeout": (x.status == 0).mean(), "med_days_to_pass": x.loc[x.status == 1, "days"].median(),
                             "windows": len(x) / x.seed.nunique()})
    return pd.concat([r.groupby(r.index.year).apply(f), f(r).to_frame("ALL").T])


def main(sym):
    c = CFG[sym]
    g = h.Grid.build([sym], "M15", costs=h.COSTS_RAW)
    rows = []
    for hour, n, sl in itertools.product(c["hours"], (0, 3, 10, 20), (0.2, 0.3, 0.4)):
        r = evaluate(g, sym, hour, n, sl, start=c["train"][0], end=c["train"][1], weekdays=c["weekdays"], lev=c["lev"])
        rows.append(dict(hour=hour, n=n, sl=sl, train_pass=(r.status == 1).mean()))
    grid = pd.DataFrame(rows).sort_values("train_pass", ascending=False)
    print(f"== {sym} train-period grid (top 10 of {len(grid)})")
    print(grid.head(10).round(3).to_string())
    best = grid.iloc[0]
    print(f"selected on train: hour={best.hour} n={int(best.n)} sl={best.sl}")
    for label, costs in (("raw costs", h.COSTS_RAW), ("conservative costs", h.COSTS)):
        g2 = g if costs is h.COSTS_RAW else h.Grid.build([sym], "M15", costs=costs)
        r = evaluate(g2, sym, best.hour, int(best.n), best.sl, start=c["train"][0], weekdays=c["weekdays"], lev=c["lev"])
        print(f"-- {sym} selected config, {label}, all years (test starts {c['test'][0][:4]})")
        print(by_year(r).round(3).to_string())
    # same config, random direction (no-edge reference)
    r0 = evaluate(g, sym, best.hour, 0, best.sl, start=c["train"][0], weekdays=c["weekdays"], lev=c["lev"])
    print(f"-- {sym} same config with coin-flip direction (no-edge reference)")
    print(by_year(r0).round(3).T.to_string())
    return best


if __name__ == "__main__":
    for s in sys.argv[1:] or ["XAUUSD", "BTCUSD"]:
        main(s)
