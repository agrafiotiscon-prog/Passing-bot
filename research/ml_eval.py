"""Turn walk-forward ML predictions into costed trades and summarize."""
import numpy as np
import pandas as pd

import harness as h
import ml
import strategies as S


def pred_to_sig15(pred: pd.Series, df15: pd.DataFrame, q: float = 0.8) -> pd.Series:
    """Trade when |pred| exceeds the previous year's q-quantile of |pred|.
    Hourly prediction known at the close of the hour -> signal on the M15 bar
    that closes that hour (minute 45)."""
    ap = pred.abs()
    thr = ap.groupby(ap.index.year).quantile(q).shift(1)
    th = thr.reindex(pred.index.year).to_numpy()
    sig = pd.Series(np.where(ap.to_numpy() > th, np.sign(pred.to_numpy()), 0.0), index=pred.index)
    sig15 = pd.Series(0.0, index=df15.index)
    last_bar = df15.index[df15.index.minute == 45]
    sig15.loc[last_bar] = np.nan_to_num(sig.reindex(last_bar.floor("h")).to_numpy())
    return sig15


def run(sym, cross, hz="y4", hold=16, sl_atr=4.0, q=0.8, model="lgb", params=None):
    hh, X, Y = ml.dataset(sym, cross)
    g = h.Grid.build([sym], "M15")
    df = g.bars[sym]
    p = ml.walk_forward(X, Y[hz], model=model, params=params)
    tr = h.to_grid(g, sym, S.signal_trades(df, pred_to_sig15(p, df, q), hold, sl_atr=sl_atr))
    return p, h.trade_stats(g, tr)
