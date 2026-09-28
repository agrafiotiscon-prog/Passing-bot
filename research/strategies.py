"""Signal generators. Each returns a DataFrame of trades:
t0 (entry bar time, filled at its open), dirn, stop, tgt, t1 (last bar held).
All decisions use only data up to the close of the bar before t0."""
import numpy as np
import pandas as pd


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def daily_bars(df: pd.DataFrame) -> pd.DataFrame:
    d = df.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    return d.dropna()


def daily_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """ATR of completed daily bars, known from the first bar of the next day."""
    d = daily_bars(df)
    a = atr(d, n).shift(1)
    return a.reindex(df.index.normalize()).set_axis(df.index)


def _mk(t0, dirn, stop, tgt, t1) -> pd.DataFrame:
    return pd.DataFrame({"t0": t0, "dirn": dirn, "stop": stop, "tgt": tgt, "t1": t1})


def next_bar_times(df: pd.DataFrame, mask: pd.Series) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Signal at close of bar i -> entry at open of bar i+1."""
    pos = np.nonzero(mask.to_numpy())[0]
    pos = pos[pos + 1 < len(df)]
    return df.index[pos + 1], pos


def random_entries(df: pd.DataFrame, per_day=2, sl_atr=0.3, rr=2.0, hold_bars=32, seed=0) -> pd.DataFrame:
    """No-edge baseline: random times/directions, daily-ATR stop, fixed RR target."""
    rng = np.random.default_rng(seed)
    a = daily_atr(df).to_numpy()
    n = len(df)
    bars_per_day = n / df.index.normalize().nunique()
    m = (rng.random(n) < per_day / bars_per_day) & ~np.isnan(a)
    m[-2:] = False
    pos = np.nonzero(m)[0]
    d = rng.choice([-1.0, 1.0], len(pos))
    ent = df["open"].to_numpy()[pos + 1]
    risk = sl_atr * a[pos]
    t1 = df.index[np.minimum(pos + 1 + hold_bars, n - 1)]
    return _mk(df.index[pos + 1], d, ent - d * risk, ent + d * risk * rr, t1)


def _hour(ix: pd.DatetimeIndex) -> np.ndarray:
    return ix.hour + ix.minute / 60.0


def range_breakout(df: pd.DataFrame, rs: float, re: float, ee: float, xh: float,
                   sl_frac: float = 1.0, rr: float = np.nan, buf: float = 0.0,
                   wmin: float = 0.0, wmax: float = 9.0, oco: bool = True,
                   side: int = 0) -> pd.DataFrame:
    """Session range breakout.
    Range = high/low of bars with server hour in [rs, re). Buy-stop at the
    range high and sell-stop at the range low (+/- buf * range) are live from
    re until ee; the position is closed at xh (server hour, same day; xh>=24
    means next day xh-24). Stop = entry -/+ sl_frac * range width.
    Range width must lie in [wmin, wmax] * daily ATR(14).
    side: 0 both, +1 long only, -1 short only."""
    h = _hour(df.index)
    day = df.index.normalize()
    a = daily_atr(df)
    inr = (h >= rs) & (h < re)
    g = df[inr].groupby(day[inr])
    rng = pd.DataFrame({"hi": g["high"].max(), "lo": g["low"].min(), "n": g.size()})
    rng["atr"] = a[inr].groupby(day[inr]).first()
    rng = rng[rng.n >= max(1, int((re - rs) * 4 * 0.75))].dropna()
    w = rng.hi - rng.lo
    rng = rng[(w >= wmin * rng.atr) & (w <= wmax * rng.atr) & (w > 0)]
    w = rng.hi - rng.lo
    d0 = rng.index
    t_start = d0 + pd.to_timedelta(re, unit="h")
    t_exp = d0 + pd.to_timedelta(ee, unit="h") - pd.Timedelta(minutes=1)
    t_exit = d0 + pd.to_timedelta(xh, unit="h") - pd.Timedelta(minutes=1)
    rows = []
    gid = np.arange(len(rng)) if oco else np.full(len(rng), -1)
    for dirn, lvl in ((1, rng.hi + buf * w), (-1, rng.lo - buf * w)):
        if side and dirn != side:
            continue
        risk = sl_frac * w
        rows.append(pd.DataFrame({
            "t0": t_start, "dirn": float(dirn), "lvl": lvl.to_numpy(),
            "stop": (lvl - dirn * risk).to_numpy(),
            "tgt": (lvl + dirn * risk * rr).to_numpy() if not np.isnan(rr) else np.nan,
            "t1": t_exit, "texp": t_exp, "grp": gid}))
    out = pd.concat(rows, ignore_index=True)
    # entry bars must exist: snap t0 to the first bar at/after t_start
    ix = df.index.as_unit("ns")
    pos = np.searchsorted(ix.asi8, pd.DatetimeIndex(out.t0).as_unit("ns").asi8)
    ok = pos < len(ix)
    out = out[ok].copy()
    out["t0"] = ix[pos[ok]]
    return out.sort_values("t0").reset_index(drop=True)


from numba import njit as _njit


@_njit(cache=True)
def _channel_loop(o, h, l, c, a, hh, ll, xh, xl, sl_atr, max_hold, side):
    n = len(c)
    out_t0 = []
    out_d = []
    out_s = []
    out_t1 = []
    pos = 0
    t_in = 0
    for i in range(1, n - 1):
        if pos != 0:
            # exit on close through the trailing channel or max hold
            if (pos > 0 and c[i] < xl[i]) or (pos < 0 and c[i] > xh[i]) or (i - t_in >= max_hold):
                out_t1.append(i)
                pos = 0
            else:
                continue
        if np.isnan(a[i]) or np.isnan(hh[i]):
            continue
        if c[i] > hh[i] and side >= 0:
            pos = 1
        elif c[i] < ll[i] and side <= 0:
            pos = -1
        if pos != 0:
            t_in = i + 1
            out_t0.append(i + 1)
            out_d.append(pos)
            out_s.append(o[i + 1] - pos * sl_atr * a[i])
    if pos != 0:
        out_t1.append(n - 1)
    return np.array(out_t0), np.array(out_d), np.array(out_s), np.array(out_t1)


def channel_trend(df: pd.DataFrame, n_entry: int, n_exit: int, sl_atr: float = 3.0,
                  atr_n: int = 56, max_hold: int = 10_000, side: int = 0) -> pd.DataFrame:
    """Donchian breakout: close beyond the prior n_entry-bar high/low enters at
    the next open; exit at the close that breaks the prior n_exit-bar
    opposite extreme; protective stop sl_atr * ATR(atr_n) from entry."""
    a = atr(df, atr_n).to_numpy()
    hh = df["high"].rolling(n_entry).max().shift(1).to_numpy()
    ll = df["low"].rolling(n_entry).min().shift(1).to_numpy()
    xh = df["high"].rolling(n_exit).max().shift(1).to_numpy()
    xl = df["low"].rolling(n_exit).min().shift(1).to_numpy()
    t0, d, s, t1 = _channel_loop(df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
                                 df["close"].to_numpy(), a, hh, ll, xh, xl, sl_atr, max_hold, side)
    ix = df.index
    return _mk(ix[t0], d.astype(float), s, np.nan, ix[t1])


@_njit(cache=True)
def _sig_loop(o, a, sig, hold, sl_atr, tgt_atr, flip):
    n = len(o)
    t0s = []
    ds = []
    ss = []
    tgs = []
    t1s = []
    busy_until = -1
    cur = 0
    for i in range(n - 1):
        s = sig[i]
        if s == 0 or np.isnan(a[i]):
            continue
        if i + 1 <= busy_until and not (flip and s != cur):
            continue
        ent = o[i + 1]
        t0s.append(i + 1)
        ds.append(s)
        ss.append(ent - s * sl_atr * a[i])
        tgs.append(ent + s * tgt_atr * a[i] if tgt_atr > 0 else np.nan)
        t1s.append(min(i + hold, n - 1))
        busy_until = i + hold
        cur = s
    return np.array(t0s), np.array(ds, np.float64), np.array(ss), np.array(tgs), np.array(t1s)


def signal_trades(df: pd.DataFrame, sig: pd.Series, hold: int, sl_atr: float,
                  tgt_atr: float = 0.0, atr_n: int = 96, flip: bool = False) -> pd.DataFrame:
    """sig: -1/0/+1 known at the close of each bar -> enter at the next open,
    hold `hold` bars, stop sl_atr*ATR(atr_n), optional target tgt_atr*ATR.
    One position at a time (an opposite signal reverses if flip)."""
    a = atr(df, atr_n).to_numpy()
    s = sig.reindex(df.index).fillna(0).to_numpy(np.float64)
    t0, d, st, tg, t1 = _sig_loop(df["open"].to_numpy(), a, s, hold, sl_atr, tgt_atr, flip)
    ix = df.index
    if len(t0) == 0:
        return _mk(ix[:0], [], [], [], ix[:0])
    return _mk(ix[t0], d, st, tg, ix[t1])


def orb_first_candle(df: pd.DataFrame, open_h: float = 16.5, bar_min: int = 5, exit_h: float = 22.9,
                     rr: float = 10.0, sl_mode: str = "candle", sl_atr: float = 0.1,
                     min_body: float = 0.0) -> pd.DataFrame:
    """Zarattini & Aziz (2023) style ORB: direction of the first `bar_min`
    candle after the cash open (server hour open_h); enter at the next bar's
    open, stop at the candle's opposite extreme (or sl_atr*daily ATR), target
    rr*risk, flat at exit_h."""
    h = _hour(df.index)
    a = daily_atr(df)
    first = np.isclose(h, open_h)
    pos = np.nonzero(first)[0]
    pos = pos[pos + 1 < len(df)]
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    body = c[pos] - o[pos]
    ok = np.abs(body) > min_body * a.to_numpy()[pos]
    pos = pos[ok & (body != 0)]
    d = np.sign(c[pos] - o[pos])
    ent = o[pos + 1]
    if sl_mode == "candle":
        stop = np.where(d > 0, lo[pos], hi[pos])
    else:
        stop = ent - d * sl_atr * a.to_numpy()[pos]
    risk = np.abs(ent - stop)
    good = risk > 0
    pos, d, ent, stop, risk = pos[good], d[good], ent[good], stop[good], risk[good]
    day = df.index[pos].normalize()
    t1 = day + pd.to_timedelta(exit_h, unit="h")
    return _mk(df.index[pos + 1], d, stop, ent + d * rr * risk, t1)


def daily_fixed_time(df: pd.DataFrame, hour: float, dirn, sl_atr: float, rr: float,
                     exit_h: float = 23.5, days_mask=None) -> pd.DataFrame:
    """One market entry per day at server `hour`; dirn is +1/-1, or a Series
    indexed by day (values -1/0/+1), or 'random' (seeded coin flip)."""
    h = _hour(df.index)
    a = daily_atr(df).to_numpy()
    pos = np.nonzero(np.isclose(h, hour))[0]
    pos = pos[~np.isnan(a[pos])]
    day = df.index[pos].normalize()
    if isinstance(dirn, str):
        rng = np.random.default_rng(int(dirn.split(":")[1]) if ":" in dirn else 0)
        d = rng.choice([-1.0, 1.0], len(pos))
    elif isinstance(dirn, pd.Series):
        d = dirn.reindex(day).fillna(0).to_numpy(float)
    else:
        d = np.full(len(pos), float(dirn))
    keep = d != 0
    if days_mask is not None:
        keep &= days_mask.reindex(day).fillna(False).to_numpy(bool)
    pos, d, day = pos[keep], d[keep], day[keep]
    ent = df["open"].to_numpy()[pos]
    risk = sl_atr * a[pos]
    t1 = day + pd.to_timedelta(exit_h, unit="h")
    return _mk(df.index[pos], d, ent - d * risk, ent + d * rr * risk if rr > 0 else np.nan, t1)


def sweep_reversal(df: pd.DataFrame, start_h: float = 3.0, end_h: float = 20.0, rr: float = 2.0,
                   buf_atr: float = 0.02, exit_h: float = 22.75, tgt_mode: str = "rr") -> pd.DataFrame:
    """Liquidity sweep / turtle-soup intraday: a bar trades beyond the prior
    day's high (low) and CLOSES back inside -> short (long) at the next open,
    stop just beyond the bar's extreme, target rr*risk or the opposite side of
    the prior day's range. One trade per side per day."""
    h = _hour(df.index)
    day = df.index.normalize()
    D = daily_bars(df)
    ph = D["high"].shift(1).reindex(day).to_numpy()
    pl = D["low"].shift(1).reindex(day).to_numpy()
    a = daily_atr(df).to_numpy()
    hi, lo, cl, op = (df[c].to_numpy() for c in ("high", "low", "close", "open"))
    win = (h >= start_h) & (h < end_h)
    short = win & (hi > ph) & (cl < ph)
    long_ = win & (lo < pl) & (cl > pl)
    rows = []
    for mask, d in ((short, -1.0), (long_, 1.0)):
        pos = np.nonzero(mask)[0]
        pos = pos[pos + 1 < len(df)]
        if len(pos) == 0:
            continue
        first = ~pd.Series(day[pos]).duplicated().to_numpy()   # first signal per day per side
        pos = pos[first]
        ent = op[pos + 1]
        stop = np.where(d < 0, hi[pos], lo[pos]) - d * buf_atr * a[pos]
        risk = np.abs(ent - stop)
        tgt = np.where(d < 0, pl[pos], ph[pos]) if tgt_mode == "range" else ent + d * rr * risk
        t1 = day[pos] + pd.to_timedelta(exit_h, unit="h")
        rows.append(_mk(df.index[pos + 1], np.full(len(pos), d), stop, tgt, t1))
    out = pd.concat(rows, ignore_index=True)
    ok = (out.stop - (out.tgt)).abs() > 0
    return out[ok].sort_values("t0").reset_index(drop=True)


def nfp_drift(df: pd.DataFrame, rel_h: float = 15.5, first_min: int = 15, sl_atr: float = 0.5, rr: float = 3.0,
              exit_h: float = 22.75) -> pd.DataFrame:
    """US payrolls (first Friday of the month, 08:30 ET ~ 15:30 server):
    trade in the direction of the first `first_min` minutes after release."""
    h = _hour(df.index)
    day = df.index.normalize()
    is_nfp = (df.index.dayofweek == 4) & (df.index.day <= 7)
    a = daily_atr(df).to_numpy()
    op, cl = df["open"].to_numpy(), df["close"].to_numpy()
    bar_min = int(round((df.index[1] - df.index[0]).total_seconds() / 60))
    nb = max(1, first_min // bar_min)
    pos0 = np.nonzero(is_nfp & np.isclose(h, rel_h))[0]
    pos0 = pos0[pos0 + nb < len(df)]
    d = np.sign(cl[pos0 + nb - 1] - op[pos0])
    keep = d != 0
    pos0, d = pos0[keep], d[keep]
    e = pos0 + nb
    ent = op[e]
    risk = sl_atr * a[pos0]
    return _mk(df.index[e], d, ent - d * risk, ent + d * rr * risk, day[e] + pd.to_timedelta(exit_h, unit="h"))
