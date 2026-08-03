import pandas as pd
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


data.to_csv("data.csv")


ticker_counts = (data.groupby("Ticker").size().sort_values())

print(data.isna().sum())
