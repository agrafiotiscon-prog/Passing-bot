"""Glue between strategies (pandas) and the numba challenge simulator."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import sim
from data import load

# spread (full, price units) and slippage per fill, deliberately on the
# conservative side of what MT5 prop servers quote.
COSTS = {
    "XAUUSD": (0.30, 0.10),
    "EURUSD": (0.00010, 0.00003),
    "GBPUSD": (0.00014, 0.00004),
    "AUDUSD": (0.00012, 0.00003),
    "USDCAD": (0.00016, 0.00004),
    "USDJPY": (0.012, 0.004),
    "NAS100": (1.5, 0.5),
    "US30": (3.0, 1.0),
    "SPX500": (0.6, 0.2),
    "BTCUSD": (0.0006, 0.0002, "rel"),
}


# raw-spread prop/ECN feed without commission (the user's demo setup):
# MT5 exports show gold spreads of $0.04-0.13 median; FX majors 0.1-0.5 pip.
COSTS_RAW = {
    "XAUUSD": (0.15, 0.05),
    "EURUSD": (0.00005, 0.00001),
    "GBPUSD": (0.00008, 0.00002),
    "AUDUSD": (0.00008, 0.00002),
    "USDCAD": (0.00010, 0.00002),
    "USDJPY": (0.007, 0.002),
    "NAS100": (1.0, 0.3),
    "US30": (2.0, 0.6),
    "SPX500": (0.4, 0.1),
    "BTCUSD": (0.0003, 0.0001, "rel"),
}


# round-2 instruments (raw-spread demo assumptions)
EXTRA_RAW = {
    "AUDUSD": (0.00008, 0.00002), "USDCAD": (0.00010, 0.00002), "USDCHF": (0.00010, 0.00002),
    "AUDJPY": (0.012, 0.003), "EURCHF": (0.00012, 0.00003), "EURGBP": (0.00008, 0.00002),
    "EURJPY": (0.012, 0.003), "GBPJPY": (0.018, 0.004), "XAGUSD": (0.02, 0.005),
    "UKOIL": (0.03, 0.01), "GER40": (1.5, 0.5),
    **{c: (0.0010, 0.0002, "rel") for c in ("ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD",
                                           "LTCUSD", "LINKUSD", "BNBUSD")},
}
for _k, _v in EXTRA_RAW.items():
    COSTS_RAW.setdefault(_k, _v)
    COSTS.setdefault(_k, tuple(x * 2 if isinstance(x, float) else x for x in _v))


@dataclass
class Grid:
    syms: list
    tf: str
    idx: pd.DatetimeIndex
    O: np.ndarray
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    day: np.ndarray
    HC: np.ndarray   # half-spread per bar/instrument (price units)
    SL: np.ndarray   # slippage per fill per bar/instrument (price units)
    bars: dict = field(default_factory=dict)

    @classmethod
    def build(cls, syms, tf="M15", cost_mult=1.0, costs=None):
        costs = costs or COSTS
        bars = {s: load(s, tf) for s in syms}
        idx = bars[syms[0]].index
        for s in syms[1:]:
            idx = idx.union(bars[s].index)
        idx = idx.as_unit("ns")
        T, K = len(idx), len(syms)
        arr = {c: np.full((T, K), np.nan) for c in "OHLC"}
        for k, s in enumerate(syms):
            b = bars[s].set_axis(bars[s].index.as_unit("ns")).reindex(idx)
            arr["O"][:, k] = b["open"].to_numpy()
            arr["H"][:, k] = b["high"].to_numpy()
            arr["L"][:, k] = b["low"].to_numpy()
            arr["C"][:, k] = b["close"].to_numpy()
        day = (idx.normalize().asi8 // 86_400_000_000_000).astype(np.int64)
        HC = np.empty((T, K))
        SL = np.empty((T, K))
        for k, s in enumerate(syms):
            spec = costs[s]
            if len(spec) > 2 and spec[2] == "rel":
                px = pd.Series(arr["C"][:, k]).ffill().bfill().to_numpy()
                HC[:, k] = spec[0] / 2 * px * cost_mult
                SL[:, k] = spec[1] * px * cost_mult
            else:
                HC[:, k] = spec[0] / 2 * cost_mult
                SL[:, k] = spec[1] * cost_mult
        return cls(syms, tf, idx, arr["O"], arr["H"], arr["L"], arr["C"], day, HC, SL, bars)

    def starts(self, start=None, end=None, days=14, weekdays_only=True):
        """First bar of every (week)day in [start, end) and the last bar
        inside the following `days` calendar days."""
        ix = self.idx
        first = np.r_[True, self.day[1:] != self.day[:-1]]
        cand = np.nonzero(first)[0]
        dts = ix[cand]
        m = dts.dayofweek < 5 if weekdays_only else np.ones(len(dts), bool)
        if start is not None:
            m &= dts >= pd.Timestamp(start)
        if end is not None:
            m &= dts < pd.Timestamp(end)
        cand = cand[m]
        stop_times = ix[cand].normalize() + pd.Timedelta(days=days)
        ends = np.searchsorted(ix.asi8, stop_times.as_unit("ns").asi8) - 1
        ok = stop_times <= ix[-1]
        return cand[ok].astype(np.int64), ends[ok].astype(np.int64)


@dataclass
class Trades:
    tk: np.ndarray
    t0: np.ndarray
    dirn: np.ndarray
    stop: np.ndarray
    tgt: np.ndarray
    t1: np.ndarray
    ent: np.ndarray = None

    def __post_init__(self):
        if self.ent is None:
            self.ent = np.full(len(self.t0), np.nan)

    def __len__(self):
        return len(self.t0)

    @classmethod
    def empty(cls):
        z = np.zeros(0)
        zi = np.zeros(0, np.int64)
        return cls(zi, zi, z, z, z, zi, z)

    @classmethod
    def concat(cls, parts):
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        cat = {f: np.concatenate([getattr(p, f) for p in parts]) for f in
               ("tk", "t0", "dirn", "stop", "tgt", "t1", "ent")}
        o = np.lexsort((cat["tk"], cat["t0"]))
        return cls(**{f: v[o] for f, v in cat.items()})


def _bars(g: Grid, col, side="left"):
    return np.searchsorted(g.idx.asi8, pd.DatetimeIndex(col).as_unit("ns").asi8, side=side)


def to_grid(g: Grid, sym: str, df: pd.DataFrame) -> Trades:
    """df columns: t0 (Timestamp of entry bar), dirn, stop, tgt, t1 (Timestamp
    of last bar held). Optional stop-entry columns: lvl (entry level, nan for
    market at open of t0), texp (last bar the order may fill), grp (OCO id).
    Times must exist in the symbol's bars."""
    if len(df) == 0:
        return Trades.empty()
    k = g.syms.index(sym)
    n = len(g.idx)
    t0 = _bars(g, df["t0"])
    t1 = _bars(g, df["t1"], "right") - 1
    ent = np.full(len(df), np.nan)
    if "lvl" in df:
        lvl = df["lvl"].to_numpy(float)
        texp = np.minimum(_bars(g, df["texp"], "right") - 1, n - 1).astype(np.int64)
        grp = df["grp"].to_numpy(np.int64) if "grp" in df else np.full(len(df), -1, np.int64)
        pend = ~np.isnan(lvl)
        tkk = np.full(len(df), k, np.int64)
        fb, fp = sim.resolve_pending(g.O, g.H, g.L, tkk, np.minimum(t0, n - 1).astype(np.int64),
                                     texp, lvl, df["dirn"].to_numpy(float), np.where(pend, grp, -1))
        t0 = np.where(pend, fb, t0)
        ent = np.where(pend, fp, np.nan)
        ok_p = ~pend | (fb >= 0)
    else:
        ok_p = np.ones(len(df), bool)
    ok = ok_p & (t0 >= 0) & (t0 < n) & (t1 >= t0)
    tr = Trades(np.full(ok.sum(), k, np.int64), t0[ok].astype(np.int64),
                df["dirn"].to_numpy(float)[ok], df["stop"].to_numpy(float)[ok],
                df["tgt"].to_numpy(float)[ok], t1[ok].astype(np.int64), ent[ok])
    return Trades.concat([tr])


def challenges(g: Grid, tr: Trades, p, start=None, end=None, days=14):
    s, e = g.starts(start, end, days)
    out = sim.run_all(s, e, g.O, g.H, g.L, g.C, g.day, g.HC, g.SL,
                      tr.tk, tr.t0, tr.dirn, tr.stop, tr.tgt, tr.t1, tr.ent, p)
    res = pd.DataFrame(out, columns=["status", "end_bar", "final", "ntr", "mineq"])
    res.index = g.idx[s]
    res["days"] = (g.idx[res.end_bar.astype(int)] - res.index).total_seconds() / 86400
    return res


def summarize(res: pd.DataFrame, by_year=True) -> pd.DataFrame:
    def agg(r):
        return pd.Series({
            "n": len(r),
            "pass": (r.status == 1).mean(),
            "fail": (r.status == -1).mean(),
            "timeout": (r.status == 0).mean(),
            "days_to_pass": r.loc[r.status == 1, "days"].median(),
            "trades": r.ntr.mean(),
        })
    tot = agg(res).to_frame("ALL").T
    if not by_year:
        return tot
    yr = res.groupby(res.index.year).apply(agg)
    return pd.concat([yr, tot])


def trade_stats(g: Grid, tr: Trades) -> pd.DataFrame:
    R, xb = sim.trade_R(g.O, g.H, g.L, g.C, g.HC, g.SL,
                        tr.tk, tr.t0, tr.dirn, tr.stop, tr.tgt, tr.t1, tr.ent)
    df = pd.DataFrame({"R": R, "sym": np.array(g.syms)[tr.tk]}, index=g.idx[tr.t0])
    return df.dropna()


def r_summary(df: pd.DataFrame) -> pd.DataFrame:
    def agg(r):
        daily = r.R.groupby(r.index.normalize()).sum()
        if len(daily) > 1:
            daily = daily.reindex(pd.bdate_range(daily.index[0], daily.index[-1]), fill_value=0.0)
        return pd.Series({
            "n": len(r), "win": (r.R > 0).mean(), "avgR": r.R.mean(),
            "sumR": r.R.sum(),
            "sharpe_d": daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 2 else np.nan,
        })
    yr = df.groupby(df.index.year).apply(agg)
    return pd.concat([yr, agg(df).to_frame("ALL").T])
