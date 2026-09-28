"""Broad anomaly scan with discovery / validation / out-of-sample splits.

Rule: on each trading day, at server hour `e` (enter at the open of that
hour), if condition C holds, take direction D and hold H hours (exit at the
close of hour e+H-1). One trade per rule per day. Returns are net of the
harness cost model.

Discovery 2012-2019 -> strict filter; validation 2020-2022; OOS 2023-2026.
"""
import itertools
import sys

import numpy as np
import pandas as pd

from data import load
from harness import COSTS, COSTS_RAW

DISC = (2012, 2019)
VAL = (2020, 2022)
OOS = (2023, 2026)


def hourly(sym):
    d = load(sym, "M15")
    return d.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def day_matrix(h: pd.DataFrame):
    """Open and close per (day, hour) matrices; NaN where missing."""
    day = h.index.normalize()
    hr = h.index.hour
    O = h["open"].groupby([day, hr]).first().unstack()
    C = h["close"].groupby([day, hr]).last().unstack()
    O = O.reindex(columns=range(24))
    C = C.reindex(columns=range(24))
    return O, C


def scan(sym: str, costs=COSTS) -> pd.DataFrame:
    h = hourly(sym)
    O, C = day_matrix(h)
    days = O.index
    spread, slip = costs[sym]
    lc = np.log(h["close"])
    r1 = lc.diff()
    vol24 = r1.rolling(24).std()
    vol240 = r1.rolling(240, min_periods=120).std()
    # previous-day return (close-to-close of server days)
    dclose = h["close"].groupby(days.__class__(h.index.normalize())).last()
    dopen = h["open"].groupby(h.index.normalize()).first()
    prev_day = np.sign(np.log(dclose / dopen)).shift(1).reindex(days)
    dow = pd.Series(days.dayofweek, index=days)
    rows = []
    for e in range(24):
        # state known at the open of hour e = close of hour e-1
        t_sig = days + pd.Timedelta(hours=e) - pd.Timedelta(hours=1)
        st = pd.DataFrame(index=days)
        for L in (1, 4, 8, 24):
            x = (lc - lc.shift(L)).reindex(t_sig).to_numpy()
            st[f"p{L}"] = np.sign(x)
        tod = np.log(C.iloc[:, max(e - 1, 0)] / O.iloc[:, 0]) if e > 0 else pd.Series(np.nan, index=days)
        st["tod"] = np.sign(tod.to_numpy())
        st["pday"] = prev_day.to_numpy()
        vr = (vol24 / vol240).reindex(t_sig).to_numpy()
        st["hv"] = np.where(np.isnan(vr), np.nan, np.where(vr > 1.0, 1.0, -1.0))
        conds = {"all": np.ones(len(days), bool)}
        for c in ("p1", "p4", "p8", "p24", "tod", "pday", "hv"):
            conds[f"{c}+"] = (st[c] > 0).to_numpy()
            conds[f"{c}-"] = (st[c] < 0).to_numpy()
        for dw in range(5):
            conds[f"dow{dw}"] = (dow == dw).to_numpy()
        ent = O.iloc[:, e]
        for H in (1, 2, 3, 4, 6, 8):
            xh = e + H - 1
            if xh > 23:
                continue
            ex = C.iloc[:, xh]
            r = np.log(ex / ent).to_numpy()
            cost = (spread + 2 * slip) / ent.to_numpy()
            yrs = days.year.to_numpy()
            for cname, m in conds.items():
                ok = m & ~np.isnan(r)
                if ok.sum() < 200:
                    continue
                for D in (1, -1):
                    net = D * r[ok] - cost[ok]
                    y = yrs[ok]
                    res = {"sym": sym, "e": e, "H": H, "cond": cname, "D": D}
                    for tag, (a, b) in (("d", DISC), ("v", VAL), ("o", OOS)):
                        mm = (y >= a) & (y <= b)
                        x = net[mm]
                        res[f"{tag}_n"] = len(x)
                        res[f"{tag}_bp"] = x.mean() * 1e4 if len(x) else np.nan
                        res[f"{tag}_t"] = x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 20 else np.nan
                        if tag == "d":
                            yy = pd.Series(x).groupby(y[mm]).mean()
                            res["d_yrs_pos"] = (yy > 0).mean()
                    rows.append(res)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    raw = "--raw" in sys.argv
    syms = [a for a in sys.argv[1:] if not a.startswith("--")] or ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY"]
    out = pd.concat([scan(s, COSTS_RAW if raw else COSTS) for s in syms], ignore_index=True)
    out.to_parquet("scan_results_raw.parquet" if raw else "scan_results.parquet")
    out = out[~out.e.isin([23, 0, 1])]  # rollover hours: bid-quote spread artifacts
    print("rules tested:", len(out))
    disc = out[(out.d_t > 3.5) & (out.d_yrs_pos >= 0.75)]
    print("pass discovery:", len(disc))
    print(disc.sort_values("d_t", ascending=False).round(2).to_string())
