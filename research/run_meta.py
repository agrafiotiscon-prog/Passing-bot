"""Selection report + walk-forward meta-portfolio + challenge pass rates."""
import warnings

import numpy as np
import pandas as pd
from numba import njit

import library as Lb
import meta

warnings.filterwarnings("ignore")
SCR = meta.SCR
GROUPS = {"fx": (["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY", "GBPJPY", "AUDUSD", "USDCAD",
                  "USDCHF", "AUDJPY", "EURCHF", "XAGUSD", "UKOIL"], "2019-12-31", "2020-01-01"),
          "btc": (["BTCUSD"], "2020-12-31", "2021-01-01"),
          "alts": (["ETHUSD", "LTCUSD", "BNBUSD", "ADAUSD"], "2020-12-31", "2021-01-01")}


@njit(cache=True)
def chal(r, lam, halt, days):
    """Daily-close challenge: loss per day capped at the halt (EA flattens),
    pass/fail judged on closes only (intraday passes ignored)."""
    n = len(r)
    out = np.zeros(n)
    for s in range(n - days):
        eq = 0.0
        st = 0
        for t in range(s, s + days):
            x = r[t] * lam
            if x < -halt:
                x = -halt - 0.001
            eq += x
            if eq >= 0.10:
                st = 1
                break
            if eq <= -0.06:
                st = -1
                break
        out[s] = st
    return out


def sh(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(252) if len(x) > 50 else np.nan


if __name__ == "__main__":
    allS = pd.read_pickle(f"{SCR}/streams_v1.pkl")
    tabs, nets = [], {}
    for g, (syms, tre, tes) in GROUPS.items():
        t, n = Lb.select_and_test({s: allS[s] for s in syms}, tre, tes)
        t["group"] = g
        tabs.append(t)
        nets.update(n)
    T = pd.concat(tabs, ignore_index=True)
    pd.to_pickle((T, nets), f"{SCR}/select_v2.pkl")
    sel = T[T.selected]
    print(f"one-shot selection: {len(sel)}/{len(T)} rules; OOS>0 {np.mean(sel.test_sh > 0):.2f}; "
          f"mean OOS Sharpe {sel.test_sh.mean():.2f} (all rules {T.test_sh.mean():.2f})")
    M = meta.matrix()
    for K, L, ms in ((10, 2, 0.5), (20, 3, 0.5), (40, 3, 0.3)):
        p, picks = meta.wf(M, K, L, ms)
        p = p / p.std()
        st = pd.Series(chal(p.to_numpy(), 0.03, 0.02, 14), index=p.index)[:-14]
        print(f"WF top{K}/{L}y: OOS Sharpe {sh(p):.2f} by yr {p.groupby(p.index.year).apply(sh).round(1).to_dict()}")
        print(f"   pass@3%vol {np.mean(st == 1):.3f} by yr {st.groupby(st.index.year).apply(lambda x: (x == 1).mean()).round(2).to_dict()}")
        trail = (p.rolling(60).mean() / p.rolling(60).std() * np.sqrt(252)).shift(1).reindex(st.index)
        m = trail > 2
        print(f"   start only if trailing-60d Sharpe>2 ({m.mean():.2f} of days): pass {np.mean(st[m] == 1):.3f} by yr "
              f"{st[m].groupby(st[m].index.year).apply(lambda x: (x == 1).mean()).round(2).to_dict()}")
        if K == 10:
            for Y in (2024, 2025, 2026):
                print("     picks", Y, [k[:2] if k[0] != "PAIR" else k[:4] for k in picks[Y]])
