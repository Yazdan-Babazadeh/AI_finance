import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from tickers import tickers
benchmark = "SPY"


start_date = "2015-01-01"
end_date = "2026-01-01"


raw_data = yf.download(tickers,start=start_date,end=end_date,auto_adjust=True,group_by="columns",progress = True)


spy_data = yf.download("SPY",start=start_date,end=end_date,auto_adjust=True,progress=False)


raw_data.to_csv("stock_data.csv")
spy_data.to_csv("spy_data.csv")




close = raw_data.xs("Close",axis=1,level="Price")

volume = raw_data.xs("Volume",axis=1,level="Price")



close_long = (
    close
    .rename_axis(index="Date", columns="Ticker")
    .stack()
    .rename("Close")
    .reset_index()
)

volume_long = (
    volume
    .rename_axis(index="Date", columns="Ticker")
    .stack()
    .rename("Volume")
    .reset_index()
)


data = close_long.merge(volume_long,on=["Date","Ticker"],how="inner")

data = data.sort_values(["Ticker","Date"]).reset_index(drop = True)
data["Date"] = pd.to_datetime(data["Date"])

data["DailyReturn"] = (data.groupby("Ticker")["Close"].pct_change())


data["FutureReturn20"] = ((data.groupby("Ticker")["Close"].shift(-20)/data["Close"])-1)

data.to_csv("data.csv")


