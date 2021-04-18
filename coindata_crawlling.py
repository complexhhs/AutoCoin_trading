#%%
import pandas as pd
import pyupbit
import time
import os

if not os.path.isdir("./coin_data"):
    os.makedirs("./coin_data")

tickers = pyupbit.get_tickers(fiat="KRW")[:30]


def get_ticker_data(ticker):
    date = None
    dfs = []
    for i in range(500):
        try:
            df = pyupbit.get_ohlcv(ticker=ticker, interval="minute30", to=date)
            date = df.index[0]
            time.sleep(0.1)
        except:
            break
        df["target"] = (df["high"] + df["low"]) * 0.5
        dfs.append(df)

    df = pd.concat(dfs).sort_index()
    writer = pd.ExcelWriter(
        os.path.join(
            os.path.curdir, "coin_data", str(ticker)[4:] + "_data.xlsx"
        ),
        engine="xlsxwriter",
    )
    df.to_excel(writer)
    writer.close()
    return df


for t in tickers:
    print("")
    print(t + " coin data gathering...")
    df = get_ticker_data(t)
    print("Okay this coin data successfully gathered!")

print(len(df))
print(df.head())
