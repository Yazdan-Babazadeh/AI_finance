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

#Preparing SPY data


spy_close = spy_data["Close"].squeeze()
spy = (spy_close.rename("SPYClose").reset_index())

spy["Date"] = pd.to_datetime(spy["Date"])


spy["SPYReturn20"] = (spy["SPYClose"].shift(-20) / spy["SPYClose"]  - 1)

data = data.merge(spy[["Date","SPYClose","SPYReturn20"]],on="Date",how="left")


data["NetReturn20"] = data["FutureReturn20"] - data["SPYReturn20"]

print(data)


#Focusing on 4 factors, Momentum, Volatility, Volume Ratio, Short term Return

### Momentum

grouped_close = data.groupby("Ticker")["Close"]

data["Momentum60_5"] = grouped_close.shift(5)/grouped_close.shift(60) -1














