"""Strategy library x instrument grid, as daily P&L streams.

Every stream is the daily net return (fraction of notional, 1x position) of
one rule on one instrument. Rules use only information available before the
position is taken. Costs = raw spread + 2 x slippage per round trip, charged
per unit of position change.

Selection happens per instrument on the TRAIN span only; the OOS span is
untouched until the final report (see select_and_test)."""
import numpy as np
import pandas as pd

from data import load
from harness import COSTS_RAW

SYNTHETIC = {"EURGBP", "EURJPY", "GBPJPY"}
CRYPTO = {"BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "LTCUSD", "LINKUSD", "BNBUSD"}

EXTRA_COSTS = {
    "AUDUSD": (0.00008, 0.00002), "USDCAD": (0.00010, 0.00002), "USDCHF": (0.00010, 0.00002),
    "AUDJPY": (0.012, 0.003), "EURCHF": (0.00012, 0.00003), "EURGBP": (0.00008, 0.00002),
    "EURJPY": (0.012, 0.003), "GBPJPY": (0.018, 0.004), "XAGUSD": (0.02, 0.005),
    "UKOIL": (0.03, 0.01), "GER40": (1.5, 0.5),
    **{c: (0.0010, 0.0002, "rel") for c in CRYPTO - {"BTCUSD"}},
}


def cost_rt(sym: str, px: pd.Series) -> pd.Series:
    spec = COSTS_RAW.get(sym) or EXTRA_COSTS[sym]
    c = spec[0] + 2 * spec[1]
    return pd.Series(c, index=px.index) if len(spec) > 2 and spec[2] == "rel" else c / px


def bars(sym: str):
    """Hourly and daily bars. For non-crypto symbols the rollover window
    (server 23:00-02:00, incl. gold's daily break and re-open) is dropped:
    bid bars there carry blown-out spreads (artificial lows / opens), which
    created fake edges."""
    d = load(sym, "M15")
    if sym not in CRYPTO:
        hm = d.index.hour * 60 + d.index.minute
        d = d[(hm >= 120) & (hm < 23 * 60)]
    h = d.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    D = d.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    if sym not in CRYPTO:
        D = D[D.index.dayofweek < 5]
    # drop days with too little data
    cnt = d.groupby(d.index.normalize()).size()
    D = D[cnt.reindex(D.index).fillna(0) >= 40]
    return h, D


def _rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def _pos_pnl(pos: pd.Series, r: pd.Series, c: pd.Series):
    """pos decided at the previous close, held over day d's close-to-close.
    Returns (gross, cost)."""
    pos = pos.fillna(0)
    turn = pos.diff().abs().fillna(pos.abs())
    return pos * r, turn * c / 2


def _day_trade(sig: pd.Series, ret: pd.Series, c: pd.Series):
    """Enter and exit within the day. Returns (gross, cost)."""
    s = sig.fillna(0).where(ret.notna(), 0.0)
    return s * ret.fillna(0), s.abs() * c


def streams(sym: str) -> pd.DataFrame:
    h, D = bars(sym)
    C, O, Hh, Ll = D.close, D.open, D.high, D.low
    r = np.log(C / C.shift())
    c = cost_rt(sym, C)
    out = {}
    lc = np.log(C)
    # 1 time-series momentum
    for L in (2, 5, 10, 20, 40, 60, 120):
        out[f"tsmom{L}"] = _pos_pnl(np.sign(lc - lc.shift(L)).shift(1), r, c)
    # 2 EMA crossovers
    for f, s in ((5, 20), (10, 50), (20, 100)):
        out[f"ema{f}_{s}"] = _pos_pnl(np.sign(C.ewm(span=f).mean() - C.ewm(span=s).mean()).shift(1), r, c)
    # 3 Donchian channel (stay in until opposite break)
    for N in (10, 20, 55):
        up = C > Hh.rolling(N).max().shift(1)
        dn = C < Ll.rolling(N).min().shift(1)
        st = pd.Series(np.where(up, 1.0, np.where(dn, -1.0, np.nan)), index=C.index).ffill()
        out[f"donch{N}"] = _pos_pnl(st.shift(1), r, c)
    # 4 RSI(2) / IBS / streak mean reversion (one-day holds)
    r2 = _rsi(C, 2)
    for th in (5, 10, 20):
        sig = pd.Series(np.where(r2 < th, 1.0, np.where(r2 > 100 - th, -1.0, 0.0)), index=C.index)
        out[f"rsi2_{th}"] = _pos_pnl(sig.shift(1), r, c)
    ibs = (C - Ll) / (Hh - Ll)
    for th in (0.1, 0.2, 0.3):
        sig = pd.Series(np.where(ibs < th, 1.0, np.where(ibs > 1 - th, -1.0, 0.0)), index=C.index)
        out[f"ibs{th}"] = _pos_pnl(sig.shift(1), r, c)
    dnd = (r < 0).astype(int)
    upd = (r > 0).astype(int)
    for k in (2, 3):
        sig = pd.Series(0.0, index=C.index)
        sig[dnd.rolling(k).sum() == k] = 1.0
        sig[upd.rolling(k).sum() == k] = -1.0
        out[f"streak{k}"] = _pos_pnl(sig.shift(1), r, c)
    # intraday building blocks from hourly bars
    day = h.index.normalize()
    hr = h.index.hour
    Om = h["open"].groupby([day, hr]).first().unstack().reindex(columns=range(24))
    Cm = h["close"].groupby([day, hr]).last().unstack().reindex(columns=range(24))
    Om = Om.reindex(D.index)
    Cm = Cm.reindex(D.index)
    first_open = Om.bfill(axis=1).iloc[:, 0]
    last_close = Cm.ffill(axis=1).iloc[:, -1]
    # 5 session drifts (orientation chosen later on train)
    for a, b in ((2, 8), (8, 16), (16, 23), (2, 9), (9, 15), (15, 23), (2, 5), (13, 17), (19, 23)):
        ret = np.log(Cm[b - 1] / Om[a])
        out[f"sess{a}_{b}"] = _day_trade(pd.Series(1.0, index=D.index), ret, c)
    # 6 intraday momentum: sign(open->T) trades T->end of day
    for T in (4, 8, 12, 16, 20):
        sig = np.sign(np.log(Cm[T - 1] / first_open))
        out[f"imom{T}"] = _day_trade(sig, np.log(last_close / Cm[T - 1]), c)
    # 7 first-hour-of-session direction -> rest of session
    for S, E in ((2, 9), (10, 16), (16, 23)):
        sig = np.sign(np.log(Cm[S] / Om[S]))
        out[f"orb{S}"] = _day_trade(sig, np.log(Cm[E - 1] / Cm[S]), c)
    # 8 prior-day direction continuation into today's open->close
    pd_dir = np.sign(np.log(C / O)).shift(1)
    out["pdcont"] = _day_trade(pd_dir, np.log(last_close / first_open), c)
    # 9 day-of-week drift (close-to-close)
    for k in range(5):
        m = pd.Series((D.index.dayofweek == k).astype(float), index=D.index)
        out[f"dow{k}"] = _day_trade(m, r, c)
    # 10 Monday gap fade (open->close on Mondays / first day of week)
    gap = np.log(O / C.shift())
    first_day = pd.Series(D.index.dayofweek, index=D.index).diff().fillna(0) < 0
    out["gapfade"] = _day_trade((-np.sign(gap)).where(first_day, 0.0), np.log(C / O), c)
    # 11 volatility breakout (Williams): first touch of open +/- k*prev range, hold to close
    prng = (Hh - Ll).shift(1)
    hh_cum = h["high"].groupby(day).cummax()
    ll_cum = h["low"].groupby(day).cummin()
    for k in (0.3, 0.5, 0.8):
        up_l = (O + k * prng).reindex(day).to_numpy()
        dn_l = (O - k * prng).reindex(day).to_numpy()
        hit_up = pd.Series(hh_cum.to_numpy() >= up_l, index=h.index).groupby(day).idxmax()
        any_up = pd.Series(hh_cum.to_numpy() >= up_l, index=h.index).groupby(day).any()
        hit_dn = pd.Series(ll_cum.to_numpy() <= dn_l, index=h.index).groupby(day).idxmax()
        any_dn = pd.Series(ll_cum.to_numpy() <= dn_l, index=h.index).groupby(day).any()
        tu = pd.Series(np.where(any_up, hit_up.astype("int64"), np.iinfo(np.int64).max), index=any_up.index)
        td = pd.Series(np.where(any_dn, hit_dn.astype("int64"), np.iinfo(np.int64).max), index=any_dn.index)
        long_first = (tu < td).reindex(D.index).fillna(False)
        short_first = (td < tu).reindex(D.index).fillna(False)
        lvl = pd.Series(np.where(long_first, O + k * prng, np.where(short_first, O - k * prng, np.nan)), index=D.index)
        sig = pd.Series(np.where(long_first, 1.0, np.where(short_first, -1.0, 0.0)), index=D.index)
        out[f"vbo{k}"] = _day_trade(sig, np.log(C / lvl).fillna(0), c)
    G = pd.DataFrame({k: v[0] for k, v in out.items()})
    K = pd.DataFrame({k: v[1] for k, v in out.items()})
    if sym in SYNTHETIC:
        # high/low of synthetic crosses are product bounds -> drop H/L-based rules
        drop = [c for c in G.columns if c.startswith(("vbo", "ibs", "donch"))]
        G, K = G.drop(columns=drop), K.drop(columns=drop)
    keep = G.index >= G.index[0] + pd.Timedelta(days=200)
    return G.loc[keep], K.loc[keep]


def sharpe(x: pd.Series, ann: float = 252) -> float:
    x = x.dropna()
    if len(x) < 60 or x.std() == 0:
        return np.nan
    return x.mean() / x.std() * np.sqrt(ann)


def select_and_test(all_streams: dict, train_end: str, test_start: str, min_sh=0.5, min_t=2.0):
    """all_streams: sym -> (gross, cost). Each rule is also tried inverted
    (-gross, same cost). Keep a rule if train Sharpe >= min_sh and train
    t-stat >= min_t. Returns (table, dict of selected net OOS-and-IS streams)."""
    rows = []
    nets = {}
    for sym, (G, K) in all_streams.items():
        ann = 365 if sym in CRYPTO else 252
        for col in G.columns:
            for sgn, tag in ((1, ""), (-1, "_inv")):
                net = sgn * G[col] - K[col]
                tr = net.loc[:train_end]
                te = net.loc[test_start:]
                sh_tr = sharpe(tr, ann)
                yrs = len(tr.dropna()) / ann
                key = (sym, col + tag)
                rows.append(dict(sym=sym, strat=col + tag, train_sh=sh_tr,
                                 train_t=sh_tr * np.sqrt(max(yrs, 1e-9)) if not np.isnan(sh_tr) else np.nan,
                                 test_sh=sharpe(te, ann), test_days=int(te.dropna().shape[0]),
                                 turnover=float((K[col] > 0).mean())))
                nets[key] = net
    t = pd.DataFrame(rows)
    t["selected"] = (t.train_sh >= min_sh) & (t.train_t >= min_t)
    return t, nets


def portfolio(nets: dict, keys: list, vol_win: int = 60) -> pd.Series:
    """Equal-risk portfolio: each stream scaled by its trailing vol."""
    cols = {}
    for k in keys:
        x = nets[k]
        v = x.rolling(vol_win, min_periods=20).std().shift(1)
        cols[k] = x / v
    P = pd.DataFrame(cols)
    P.index = P.index.normalize()
    P = P.groupby(level=0).sum(min_count=1)
    return P.mean(axis=1, skipna=True), P
