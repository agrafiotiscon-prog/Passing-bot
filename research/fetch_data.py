"""Download the public historical price data used by the research backtests.

All sources are public GitHub repositories (the only data hosts reachable from
the research environment). Files land in $DATA_DIR/raw (default ./data/raw).

Coverage (as of 2026-09):
  XAUUSD  MT5 broker export H1 2009-05..2026-07, M30 2018-01..2026-07,
          M15 2022-04..2026-07 (mohammedfarhan14/APEX-JEV)
          MT5 broker export M5 2025-04..2026-09 (dineshelumalai007/xauusd-5m)
          ejtraderLabs MT5 M15/H1 2012-11..2022-08
          getdata.finance 1m/5m sample 2026-03..2026-09
  EURUSD  HistData M1 2000..2025 (asiertrades/HISDATA-EURUSD)
          getdata.finance 1m sample 2026-03..2026-09
  Indices TheSnowGuru M5/M15/H1 (Dukascopy) + getdata.finance 1m 2026 samples
  FX      ejtraderLabs m15/h1 for 11 more pairs 2012-2022
          devffex MT5 M15 EURUSD/GBPUSD/USDJPY 2022-08..2026-08
"""
import os
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parent.parent / "data")) / "raw"

GH = "https://raw.githubusercontent.com"
APEX = f"{GH}/mohammedfarhan14/APEX-JEV/main/data/raw"
EJ = f"{GH}/ejtraderLabs/historical-data/main"
SNOW = f"{GH}/TheSnowGuru/Stocks-Futures-Financial-Time-series-Tick-Bar-Data/master"
HIST = f"{GH}/asiertrades/HISDATA-EURUSD/master"
GD = f"{GH}/getdata-finance"

FILES = {
    # --- gold -------------------------------------------------------------
    "xau/apex_H1.csv": f"{APEX}/XAUUSD_H1_200905180100_202607280400.csv",
    "xau/apex_M30.csv": f"{APEX}/XAUUSD_M30_201801240100_202607280400.csv",
    "xau/apex_M15.csv": f"{APEX}/XAUUSD_M15_202204200100_202607280415.csv",
    "xau/dinesh_M5.csv": f"{GH}/dineshelumalai007/xauusd-5m/main/gold_m5_historical.csv",
    "xau/ej_M15.csv": f"{EJ}/XAUUSD/XAUUSDm15.csv",
    "xau/ej_H1.csv": f"{EJ}/XAUUSD/XAUUSDh1.csv",
    "xau/snow_M5.csv": f"{SNOW}/commodities/gold/XAUUSD_M5.csv",
    "xau/snow_M15.csv": f"{SNOW}/commodities/gold/XAUUSD_M15.csv",
    "xau/snow_H1.csv": f"{SNOW}/commodities/gold/XAUUSD_H1.csv",
    "xau/gd_1m.csv": f"{GD}/xauusd-1m-ohlcv-metals-historical-data/main/XAUUSD_1m.csv",
    "xau/gd_5m.csv": f"{GD}/xauusd-5m-ohlcv-metals-historical-data/main/XAUUSD_5m.csv",
    # --- EURUSD -----------------------------------------------------------
    **{f"eurusd/hist_M1_{y}.csv": f"{HIST}/DAT_ASCII_EURUSD_M1_{y}.csv" for y in range(2010, 2026)},
    "eurusd/gd_1m.csv": f"{GD}/eurusd-1m-ohlcv-forex-historical-data/main/EURUSD_1m.csv",
    "eurusd/snow_M5.csv": f"{SNOW}/forex/eurusd/EURUSD_M5.csv",
    # --- indices ----------------------------------------------------------
    **{f"{k}/snow_{tf}.csv": f"{SNOW}/indices/{d}/{s}_{tf}.csv"
       for k, d, s in [("nas100", "nasdaq100", "USATECHIDXUSD"),
                       ("us30", "dow30", "USA30IDXUSD"),
                       ("spx500", "s%26p500", "USA500IDXUSD"),
                       ("ger40", "dax30", "DEUIDXEUR")]
       for tf in ("M5", "M15", "H1")},
    "nas100/gd_1m.csv": f"{GD}/nas100-1m-ohlcv-index-historical-data/main/NAS100_1m.csv",
    "us30/gd_1m.csv": f"{GD}/us30-1m-ohlcv-index-historical-data/main/US30_1m.csv",
    "spx500/gd_1m.csv": f"{GD}/spx500-1m-ohlcv-index-historical-data/main/SPX500_1m.csv",
    "gbpusd/gd_1m.csv": f"{GD}/gbpusd-1m-ohlcv-forex-historical-data/main/GBPUSD_1m.csv",
    # daily index samples 2024-05..2026-09
    "nas100/gd_1d.csv": f"{GD}/nas100-1d-ohlcv-index-historical-data/main/NAS100_1d.csv",
    "spx500/gd_1d.csv": f"{GD}/spx500-1d-ohlcv-index-historical-data/main/SPX500_1d.csv",
    "us30/gd_1d.csv": f"{GD}/us30-1d-ohlcv-index-historical-data/main/US30_1d.csv",
    # --- MT5 exports 2022-08..2026-08 (devffex/dataset) ---------------------
    **{f"{p.lower()}/dev_M15.parquet": f"{GH}/devffex/dataset/main/{p}/M15/history.parquet"
       for p in ("EURUSD", "GBPUSD", "USDJPY")},
    # --- BTCUSD Bitstamp 1-min (ff137/bitstamp-btcusd-minute-data) ---------
    "btcusd/bitstamp_2012_2025.csv.gz": f"{GH}/ff137/bitstamp-btcusd-minute-data/main/data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz",
    "btcusd/bitstamp_latest.csv": f"{GH}/ff137/bitstamp-btcusd-minute-data/main/data/updates/btcusd_bitstamp_1min_latest.csv",
    # --- more instruments (round 2) ----------------------------------------
    "xagusd/snow_M15.csv": f"{SNOW}/commodities/silver/XAGUSD_M15.csv",
    "ukoil/snow_M15.csv": f"{SNOW}/commodities/brent/BRENTCMDUSD_M15.csv",
    **{f"{k}/snow_M15.csv": f"{SNOW}/crypto/{d}/{f}_M15.csv"
       for k, d, f in [("ethusd", "ethusd", "ETHUSD"), ("solusd", "sol", "SOLUSDT"), ("xrpusd", "xrp", "XRPUSDT"),
                       ("adausd", "adausdt", "ADAUSDT"), ("dogeusd", "doge", "DOGEUSDT"), ("ltcusd", "ltc", "LTCUSDT"),
                       ("linkusd", "link", "LINKUSDT"), ("bnbusd", "bnb", "BNBUSDT")]},
    **{f"{k}/gd_15m.csv": f"{GD}/{k}-15m-ohlcv-{c}-historical-data/main/{k.upper()}_15m.csv"
       for k, c in [("xagusd", "metals"), ("usoil", "commodities"), ("ukoil", "commodities"),
                    ("audusd", "forex"), ("usdcad", "forex"), ("usdchf", "forex"), ("eurjpy", "forex"),
                    ("eurgbp", "forex")]},
    "usdcad/gd_1d.csv": f"{GD}/usdcad-1d-ohlcv-forex-historical-data/main/USDCAD_1d.csv",
    "usdchf/gd_1d.csv": f"{GD}/usdchf-1d-ohlcv-forex-historical-data/main/USDCHF_1d.csv",
    # --- other FX (MT5 broker export, 2012-2022) ---------------------------
    **{f"{p.lower()}/ej_{tf}.csv": f"{EJ}/{p}/{p}{tf.lower()}.csv"
       for p in ("EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF",
                 "EURJPY", "GBPJPY", "EURGBP", "AUDJPY", "EURCHF")
       for tf in ("M15", "H1")},
}


def fetch(url: str, dest: Path) -> str:
    if dest.exists() and dest.stat().st_size > 1000:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=300) as r:
                data = r.read()
            if data.startswith(b"version https://git-lfs"):
                return "LFS pointer, skipped"
            dest.write_bytes(data)
            return f"{len(data) / 1e6:.1f} MB"
        except Exception as e:  # network hiccup: back off and retry
            err = e
            time.sleep(2 ** (attempt + 1))
    return f"FAILED: {err}"


def main() -> None:
    only = sys.argv[1:]
    for name, url in FILES.items():
        if only and not any(name.startswith(o) for o in only):
            continue
        print(f"{name:28s} {fetch(url, RAW / name)}", flush=True)


if __name__ == "__main__":
    main()
