"""Load raw downloads, convert to MT5 server time, stitch into continuous bars.

Server time is UTC+2 / UTC+3 with EU daylight-saving dates (tz Europe/Helsinki),
which is what the broker MT5 exports used here run on (verified bar-by-bar).
00:00 server is the NY 17:00 close except for the ~4 weeks a year when US and
EU DST dates differ. The daily-loss limit resets at server midnight.

Output: data/bars/{SYMBOL}_M15.parquet (and _M5 where coverage allows) with
columns open, high, low, close, spread (price units, NaN if unknown), src.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
RAW = DATA / "raw"
BARS = DATA / "bars"


SERVER_TZ = "Europe/Helsinki"


def utc_to_server(ix: pd.DatetimeIndex) -> pd.DatetimeIndex:
    ix = ix.tz_localize("UTC") if ix.tz is None else ix.tz_convert("UTC")
    return ix.tz_convert(SERVER_TZ).tz_localize(None)


def _ohlc(d: pd.DataFrame) -> pd.DataFrame:
    return d[["open", "high", "low", "close"]].astype(float)


def read_mt5_export(p: Path, point: float) -> pd.DataFrame:
    """MT5 'Export bars' tab-separated file, already in server time."""
    d = pd.read_csv(p, sep="\t")
    d.columns = [c.strip("<>").lower() for c in d.columns]
    ix = pd.to_datetime(d["date"] + " " + d["time"], format="%Y.%m.%d %H:%M:%S")
    out = _ohlc(d).set_index(ix)
    out["spread"] = d["spread"].to_numpy() * point
    return out


def read_mt5_py(p: Path, point: float) -> pd.DataFrame:
    """CSV written from MetaTrader5.copy_rates_range, server time."""
    d = pd.read_csv(p, parse_dates=["time"]).set_index("time")
    out = _ohlc(d)
    out["spread"] = d["spread"] * point
    return out


def read_ejtrader(p: Path, digits: int) -> pd.DataFrame:
    """ejtraderLabs MT5 dump: prices as integer points, server time."""
    d = pd.read_csv(p, parse_dates=["Date"]).set_index("Date")
    out = _ohlc(d) / 10 ** digits
    out["spread"] = np.nan
    return out


def read_devffex(p: Path) -> pd.DataFrame:
    """devffex/dataset parquet: unix seconds of MT5 server time."""
    d = pd.read_parquet(p)
    out = _ohlc(d).set_index(pd.to_datetime(d["time"], unit="s"))
    out["spread"] = np.nan
    return out


def read_snow(p: Path) -> pd.DataFrame:
    """TheSnowGuru (Dukascopy) tab-separated, UTC."""
    d = pd.read_csv(p, sep="\t", parse_dates=["Time"]).set_index("Time")
    d.columns = [c.lower() for c in d.columns]
    out = _ohlc(d)
    out.index = utc_to_server(out.index)
    out["spread"] = np.nan
    return out


def read_getdata(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p)
    d.index = utc_to_server(pd.DatetimeIndex(pd.to_datetime(d["datetime"], utc=True)))
    out = _ohlc(d)
    out["spread"] = np.nan
    return out


def read_histdata(p: Path) -> pd.DataFrame:
    """HistData ASCII M1: 'YYYYMMDD HHMMSS;o;h;l;c;v'. Although documented as
    EST without DST, these files follow New York local time (verified
    against MT5 exports: a fixed-EST reading is 1h late every summer)."""
    d = pd.read_csv(p, sep=";", header=None, names=["t", "open", "high", "low", "close", "v"])
    ix = pd.DatetimeIndex(pd.to_datetime(d["t"], format="%Y%m%d %H%M%S"))
    ix = ix.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
    ok = ~ix.isna()
    out = _ohlc(d[ok]).set_index(utc_to_server(ix[ok]))
    out["spread"] = np.nan
    return out


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "spread": "mean"}
    if "src" in d:
        agg["src"] = "first"
    r = d.resample(rule, label="left", closed="left").agg(agg)
    return r.dropna(subset=["open"])


def stitch(parts: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """parts in priority order (best first). Each later part only fills
    time ranges not covered by earlier parts (at day granularity)."""
    out = []
    covered = pd.DatetimeIndex([])
    for name, d in parts:
        d = d[~d.index.duplicated()].sort_index()
        days = d.index.normalize()
        keep = ~days.isin(covered)
        d = d[keep].copy()
        d["src"] = name
        out.append(d)
        covered = covered.union(days[keep].unique())
    res = pd.concat(out).sort_index()
    return res[~res.index.duplicated()]


def build_xauusd() -> dict[str, pd.DataFrame]:
    x = RAW / "xau"
    gd = read_getdata(x / "gd_1m.csv")
    dn = read_mt5_py(x / "dinesh_M5.csv", 0.01)
    apex = read_mt5_export(x / "apex_M15.csv", 0.01)
    snow = read_snow(x / "snow_M5.csv")
    ej = read_ejtrader(x / "ej_M15.csv", 2)
    m15 = stitch([
        ("mt5_dinesh", resample(dn, "15min")),
        ("mt5_apex", apex),
        ("mt5_ej", ej),
        ("getdata", resample(gd, "15min")),
        ("dukascopy", resample(snow, "15min")),
    ])
    m5 = stitch([
        ("mt5_dinesh", dn),
        ("dukascopy", snow),
        ("getdata", resample(gd, "5min")),
    ])
    return {"M15": m15, "M5": m5}


def build_eurusd() -> dict[str, pd.DataFrame]:
    e = RAW / "eurusd"
    m1 = pd.concat([read_histdata(p) for p in sorted(e.glob("hist_M1_*.csv"))])
    gd = read_getdata(e / "gd_1m.csv")
    m5 = stitch([("histdata", resample(m1, "5min")), ("getdata", resample(gd, "5min"))])
    m15 = stitch([("histdata", resample(m5[m5.src == "histdata"], "15min")),
                  ("mt5_dev", read_devffex(e / "dev_M15.parquet")),
                  ("getdata", resample(gd, "15min"))])
    return {"M5": m5, "M15": m15}


def build_index(sym: str) -> dict[str, pd.DataFrame]:
    d = RAW / sym.lower()
    snow = read_snow(d / "snow_M5.csv")
    parts = [("dukascopy", snow)]
    if (d / "gd_1m.csv").exists():
        parts.insert(0, ("getdata", resample(read_getdata(d / "gd_1m.csv"), "5min")))
    m5 = stitch(parts)
    return {"M5": m5, "M15": resample(m5, "15min")}


def build_ej_fx(sym: str) -> dict[str, pd.DataFrame]:
    d = RAW / sym.lower()
    digits = 3 if sym.endswith("JPY") else 5
    parts = [("mt5_ej", read_ejtrader(d / "ej_M15.csv", digits))]
    if (d / "dev_M15.parquet").exists():
        parts.append(("mt5_dev", read_devffex(d / "dev_M15.parquet")))
    if (d / "gd_1m.csv").exists():
        parts.append(("getdata", resample(read_getdata(d / "gd_1m.csv"), "15min")))
    if (d / "gd_15m.csv").exists():
        parts.append(("getdata", read_getdata(d / "gd_15m.csv")))
    return {"M15": stitch(parts)}


def build_btc() -> dict[str, pd.DataFrame]:
    """Bitstamp BTC/USD 1-min (UTC) from 2016 on; 24/7."""
    b = RAW / "btcusd"
    parts = []
    for f in ("bitstamp_2012_2025.csv.gz", "bitstamp_latest.csv"):
        d = pd.read_csv(b / f)
        d = d[d.timestamp >= 1451606400]  # 2016-01-01
        d.index = utc_to_server(pd.DatetimeIndex(pd.to_datetime(d["timestamp"], unit="s")))
        parts.append(_ohlc(d).assign(spread=np.nan))
    m1 = pd.concat(parts)
    m1 = m1[~m1.index.duplicated()].sort_index()
    m5 = resample(m1, "5min")
    return {"M5": m5.assign(src="bitstamp"), "M15": resample(m1, "15min").assign(src="bitstamp")}


def build_snow_plus(sym: str) -> dict[str, pd.DataFrame]:
    """SnowGuru (Dukascopy/Binance, UTC) M15 + getdata 2026 15m sample."""
    d = RAW / sym.lower()
    parts = [("dukascopy", read_snow(d / "snow_M15.csv"))]
    if (d / "gd_15m.csv").exists():
        parts.insert(0, ("getdata", read_getdata(d / "gd_15m.csv")))
    return {"M15": stitch(parts)}


def build_synthetic_cross(sym: str) -> dict[str, pd.DataFrame]:
    """EURGBP / EURJPY / GBPJPY from the majors (2012-2026). Opens and closes
    are exact; high/low are the product bounds, so ranges are overstated --
    use for close-based research only, not for stop/target simulation."""
    legs = {"EURGBP": ("EURUSD", "GBPUSD", -1), "EURJPY": ("EURUSD", "USDJPY", 1),
            "GBPJPY": ("GBPUSD", "USDJPY", 1)}[sym]
    a = load(legs[0], "M15")
    b = load(legs[1], "M15")
    j = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    out = pd.DataFrame(index=j.index)
    if legs[2] == 1:
        out["open"], out["close"] = j.open_a * j.open_b, j.close_a * j.close_b
        out["high"], out["low"] = j.high_a * j.high_b, j.low_a * j.low_b
    else:
        out["open"], out["close"] = j.open_a / j.open_b, j.close_a / j.close_b
        out["high"], out["low"] = j.high_a / j.low_b, j.low_a / j.high_b
    out = out[["open", "high", "low", "close"]]
    out["spread"] = np.nan
    parts = [("synthetic", out)]
    real = RAW / sym.lower() / "ej_M15.csv"
    if real.exists():
        parts.insert(0, ("mt5_ej", read_ejtrader(real, 3 if sym.endswith("JPY") else 5)))
    gdp = RAW / sym.lower() / "gd_15m.csv"
    if gdp.exists():
        parts.insert(0, ("getdata", read_getdata(gdp)))
    return {"M15": stitch(parts)}


def main() -> None:
    BARS.mkdir(parents=True, exist_ok=True)
    import sys
    jobs = {"XAUUSD": build_xauusd, "EURUSD": build_eurusd, "BTCUSD": build_btc}
    for s in ("NAS100", "US30", "SPX500"):
        jobs[s] = lambda s=s: build_index(s)
    jobs["GER40"] = lambda: build_index("GER40")
    for s in ("GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "AUDJPY", "EURCHF"):
        jobs[s] = lambda s=s: build_ej_fx(s)
    for s in ("XAGUSD", "UKOIL", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "DOGEUSD", "LTCUSD", "LINKUSD", "BNBUSD"):
        jobs[s] = lambda s=s: build_snow_plus(s)
    for s in ("EURGBP", "EURJPY", "GBPJPY"):
        jobs[s] = lambda s=s: build_synthetic_cross(s)
    only = sys.argv[1:]
    for sym, fn in jobs.items():
        if only and sym not in only:
            continue
        for tf, df in fn().items():
            df.to_parquet(BARS / f"{sym}_{tf}.parquet")
            print(f"{sym:7s} {tf:4s} {len(df):8d} bars {df.index[0]} -> {df.index[-1]}", flush=True)


def load(sym: str, tf: str = "M15") -> pd.DataFrame:
    return pd.read_parquet(BARS / f"{sym}_{tf}.parquet")


if __name__ == "__main__":
    main()
