"""Walk-forward meta-portfolio over all library + pairs streams."""
import numpy as np
import pandas as pd

SCR = "/tmp/claude-0/-home-user-Passing-bot/103fae14-dff5-5ce4-a680-2c69ef1de6c7/scratchpad"


def matrix():
    T, nets = pd.read_pickle(f"{SCR}/select_v2.pkl")
    pst = pd.read_pickle(f"{SCR}/pairs_v1.pkl")
    cols = {}
    for k, x in nets.items():
        cols[k] = x / x.rolling(60, min_periods=20).std().shift(1)
    for k, x in pst.items():
        cols[("PAIR",) + k] = x / x.rolling(60, min_periods=20).std().shift(1)
    M = pd.DataFrame(cols)
    M.index = M.index.normalize()
    M = M.groupby(level=0).sum(min_count=1)
    return M.replace([np.inf, -np.inf], np.nan).clip(-8, 8)


def wf(M, K=10, L=2, min_sh=0.5, first=2016, last=2026):
    res, picks_by_year = [], {}
    for Y in range(first, last + 1):
        tr = M.loc[f"{Y-L}-01-01":f"{Y-1}-12-31"]
        te = M.loc[f"{Y}-01-01":f"{Y}-12-31"]
        ok = tr.notna().sum() >= int(200 * L)
        s = tr.loc[:, ok].mean() / tr.loc[:, ok].std() * np.sqrt(252)
        s = s[s > min_sh].sort_values(ascending=False)
        picks, seen = [], set()
        for k in s.index:
            base = k if k[0] == "PAIR" else (k[0], k[1].replace("_inv", ""))
            if base in seen:
                continue
            seen.add(base)
            picks.append(k)
            if len(picks) >= K:
                break
        picks_by_year[Y] = picks
        if picks:
            res.append(te[picks].mean(axis=1, skipna=True))
    return pd.concat(res).dropna(), picks_by_year
