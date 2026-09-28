"""Build all streams for the universe, select on train, report OOS."""
import sys
import numpy as np
import pandas as pd
import library as Lb

UNIVERSE = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "EURGBP", "EURJPY", "GBPJPY",
            "AUDUSD", "USDCAD", "USDCHF", "AUDJPY", "EURCHF", "XAGUSD", "UKOIL",
            "BTCUSD", "ETHUSD", "LTCUSD", "BNBUSD", "ADAUSD", "XRPUSD", "LINKUSD", "DOGEUSD", "SOLUSD"]

if __name__ == "__main__":
    allS = {}
    for s in UNIVERSE:
        allS[s] = Lb.streams(s)
        print(s, allS[s][0].shape, flush=True)
    pd.to_pickle(allS, "/tmp/claude-0/-home-user-Passing-bot/103fae14-dff5-5ce4-a680-2c69ef1de6c7/scratchpad/streams_v1.pkl")
