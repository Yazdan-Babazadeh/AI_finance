import yfinance as yf
import pandas as pd

class DataEngine:
    def __init__(self,tickers,benchmark, start_date,end_date):

        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date

    def download_data(self):

        raw_data = yf.download(self.tickers,start = self.start_date, end= self.end_date, auto_adjust = True,group_by = "columns", progress = True)
        spy_data = yf.download(self.benchmark,start = self.start_date, end= self.end_date, auto_adjust = True, progress = False)
        
        close = raw_data.xs("Close",axis =1, level="Price")
        volume = raw_data.xs("Volume",axis=1,level="Price")

        close_long = (close.rename_axis(index="Date",columns="Ticker").stack().rename("Close").reset_index())

        volume_long = (volume.rename_axis(index="Date", columns = "Ticker").stack().rename("Volume").reset_index())

        data = close_long.merge(volume_long, on=["Date","Ticker"],how="inner")

        data = data.sort_values(["Ticker","Date"]).reset_index(drop=True)

        data["Date"] = pd.to_datetime(data["Date"])

        data["DailyReturn"] = (data.groupby("Ticker")["Close"].pct_change())

        data["FutureReturn20"] = (data.groupby("Ticker")["Close"].shift(-20)/data["Close"]-1)

        spy_close = spy_data["Close"].squeeze()

        spy = (spy_close.rename("SPYClose").reset_index())

        spy["Date"] = pd.to_datetime(spy["Date"])

        spy["SPYReturn20"] = (spy["SPYClose"].shift(-20) / spy["SPYClose"] -1)

        data = data.merge(spy[["Date","SPYClose","SPYReturn20"]],on="Date",how="left")

        data["NetReturn20"] = data["FutureReturn20"] - data["SPYReturn20"]

        return data












