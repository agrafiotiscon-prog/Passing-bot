"""Relative-value (pairs) mean reversion.

spread_t = log A_t - beta_t * log B_t, beta from a trailing OLS window;
z = (spread - rolling mean) / rolling std over `look` bars (all trailing).
Enter against |z| > z_in at the next bar, exit when |z| < z_out or after
max_hold bars. Both legs pay raw costs. P&L in units of gross notional of
leg A (leg B sized beta-neutral)."""
import itertools

import numpy as np
import pandas as pd
from numba import njit

from data import load
from harness import COSTS_RAW


def closes(sym: str, rule: str) -> pd.Series:
    d = load(sym, "M15")
    hm = d.index.hour * 60 + d.index.minute
    if not sym.endswith("USD") or sym in ("XAUUSD", "XAGUSD", "EURUSD", "GBPUSD", "AUDUSD"):
        d = d[(hm >= 120) & (hm < 23 * 60)]   # skip rollover for non-crypto
    return d["close"].resample(rule).last().dropna()


def cost_frac(sym: str, px: pd.Series) -> pd.Series:
    spec = COSTS_RAW[sym]
    c = spec[0] + 2 * spec[1]
    return pd.Series(c, index=px.index) if len(spec) > 2 and spec[2] == "rel" else c / px


@njit(cache=True)
def _loop(z, z_in, z_out, max_hold):
    n = len(z)
    pos = np.zeros(n)
    cur = 0.0
    held = 0
    for i in range(n - 1):
        zi = z[i]
        if np.isnan(zi):
            pos[i + 1] = cur
            continue
        if cur != 0.0:
            held += 1
            if abs(zi) < z_out or held >= max_hold or np.sign(zi) == cur:
                cur = 0.0
                held = 0
        if cur == 0.0 and abs(zi) > z_in:
            cur = -np.sign(zi)
            held = 0
        pos[i + 1] = cur
    return pos


def pair_stream(a: str, b: str, rule: str, look: int, beta_win: int, z_in: float, z_out: float,
                max_hold: int) -> pd.Series:
    A = closes(a, rule)
    B = closes(b, rule)
    j = pd.concat([A, B], axis=1, keys=["a", "b"]).dropna()
    la, lb = np.log(j.a), np.log(j.b)
    ra, rb = la.diff(), lb.diff()
    beta = (ra.rolling(beta_win).cov(rb) / rb.rolling(beta_win).var()).shift(1)
    spread = la - beta * lb
    z = ((spread - spread.rolling(look).mean()) / spread.rolling(look).std()).shift(0)
    pos = pd.Series(_loop(z.to_numpy(), z_in, z_out, max_hold), index=j.index)  # decided at close i, held i+1
    gross = pos * (ra - beta * rb)
    turn = pos.diff().abs().fillna(0)
    cost = turn * (cost_frac(a, j.a) + beta.abs() * cost_frac(b, j.b)) / 2
    net = (gross - cost).fillna(0)
    return net.groupby(net.index.normalize()).sum()


PAIRS = [("EURUSD", "GBPUSD"), ("XAUUSD", "XAGUSD"), ("EURUSD", "USDCHF"), ("EURJPY", "GBPJPY"),
         ("AUDUSD", "EURUSD"), ("USDJPY", "EURJPY"), ("BTCUSD", "ETHUSD"), ("NAS100", "SPX500"),
         ("US30", "SPX500"), ("GBPUSD", "EURGBP"), ("XAUUSD", "EURUSD"), ("UKOIL", "USDCAD")]


def grid():
    rows = []
    streams = {}
    for (a, b), (rule, look, bw), (zi, zo) in itertools.product(
            PAIRS, [("1h", 48, 240), ("1h", 120, 480), ("4h", 60, 180), ("1D", 20, 60), ("1D", 60, 120)],
            [(2.0, 0.5), (2.5, 0.0)]):
        mh = {"1h": 72, "4h": 60, "1D": 20}[rule]
        try:
            s = pair_stream(a, b, rule, look, bw, zi, zo, mh)
        except Exception as e:  # missing data
            print(a, b, e)
            continue
        key = (a, b, rule, look, zi)
        streams[key] = s
    return streams


if __name__ == "__main__":
    st = grid()
    pd.to_pickle(st, "/tmp/claude-0/-home-user-Passing-bot/103fae14-dff5-5ce4-a680-2c69ef1de6c7/scratchpad/pairs_v1.pkl")
    print(len(st))
