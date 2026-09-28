"""Walk-forward ML search for conditional intraday edges.

Hourly bars; features only use data up to the bar close; the model trained on
years < Y predicts year Y (expanding window), so every prediction is out of
sample."""
import numpy as np
import pandas as pd

from data import load


def hourly(sym: str) -> pd.DataFrame:
    d = load(sym, "M15")
    h = d.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    return h


def _rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def features(h: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    c = h["close"]
    lc = np.log(c)
    r1 = lc.diff()
    vol = r1.rolling(120, min_periods=60).std()
    f = pd.DataFrame(index=h.index)
    for k in (1, 2, 4, 8, 16, 24, 48, 96, 240):
        f[f"r{k}"] = (lc - lc.shift(k)) / (vol * np.sqrt(k))
    f["volr"] = r1.rolling(24).std() / r1.rolling(240, min_periods=120).std()
    f["vol"] = vol
    tr = pd.concat([h.high - h.low, (h.high - c.shift()).abs(), (h.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 48, adjust=False).mean()
    for n in (24, 120, 480):
        f[f"ema{n}"] = (c - c.ewm(span=n, adjust=False).mean()) / atr
    f["rsi14"] = _rsi(c, 14)
    day = h.index.normalize()
    g = h.groupby(day)
    dhi = g["high"].cummax()
    dlo = g["low"].cummin()
    dop = g["open"].transform("first")
    f["dpos"] = (c - dlo) / (dhi - dlo).replace(0, np.nan)
    f["dret"] = (c - dop) / atr
    f["drng"] = (dhi - dlo) / atr
    dd = h.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    prev = pd.DataFrame({"pret": (dd.close - dd.open) / (dd.high - dd.low).rolling(14).mean(),
                         "prng": (dd.high - dd.low) / (dd.high - dd.low).rolling(14).mean()}).shift(1)
    f = f.join(prev.reindex(day).set_axis(h.index))
    hi20 = h.high.rolling(120).max()
    lo20 = h.low.rolling(120).min()
    f["chpos"] = (c - lo20) / (hi20 - lo20)
    if prefix:
        f = f.add_prefix(prefix)
    return f


def dataset(sym: str, cross: list[str], horizons=(4, 8, 24)) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    h = hourly(sym)
    X = features(h)
    X["hour"] = h.index.hour
    X["dow"] = h.index.dayofweek
    for cs in cross:
        cf = features(hourly(cs), prefix=f"{cs}_")[[f"{cs}_r1", f"{cs}_r4", f"{cs}_r24", f"{cs}_ema120"]]
        X = X.join(cf, how="left")
    lc = np.log(h["close"])
    vol = X["vol"]
    Y = pd.DataFrame({f"y{k}": (lc.shift(-k) - lc) / (vol * np.sqrt(k)) for k in horizons})
    return h, X, Y


def walk_forward(X: pd.DataFrame, y: pd.Series, first_test_year: int = 2016, model="lgb",
                 params: dict | None = None) -> pd.Series:
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    pred = pd.Series(np.nan, index=X.index)
    ok = y.notna()
    years = sorted(X.index.year.unique())
    for Y_ in [yy for yy in years if yy >= first_test_year]:
        tr = ok & (X.index.year < Y_)
        te = X.index.year == Y_
        # embargo: drop the last few training rows whose label overlaps the test year
        Xtr, ytr = X[tr].iloc[:-48], y[tr].iloc[:-48].clip(-4, 4)
        if model == "lgb":
            p = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=400,
                     subsample=0.7, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)
            if params:
                p.update(params)
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, ytr)
            pred[te] = m.predict(X[te])
        else:
            mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
            m = Ridge(alpha=100.0).fit(((Xtr - mu) / sd).fillna(0), ytr)
            pred[te] = m.predict(((X[te] - mu) / sd).fillna(0))
    return pred
