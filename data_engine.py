import yfinance as yf
import pandas as pd

class DataEngine:
    def __init__(self,tickers,benchmark, start_date,end_date,forecast_horizon):

        self.tickers = tickers
        self.benchmark = benchmark
        self.start_date = start_date
        self.end_date = end_date
        self.forecast_horizon = forecast_horizon
        self.data = None

    def download_data(self):

        raw_data = yf.download(self.tickers,start = self.start_date, end= self.end_date, auto_adjust = True,group_by = "columns", progress = True)
        spy_data = yf.download(self.benchmark,start = self.start_date, end= self.end_date, auto_adjust = True, progress = False)
        
        close = raw_data.xs("Close",axis =1, level="Price")
        volume = raw_data.xs("Volume",axis=1,level="Price")

        close_long = (close.rename_axis(index="Date",columns="Ticker").stack().rename("Close").reset_index())

        volume_long = (volume.rename_axis(index="Date", columns = "Ticker").stack().rename("Volume").reset_index())

        self.data = close_long.merge(volume_long, on=["Date","Ticker"],how="inner")

        self.data = self.data.sort_values(["Ticker","Date"]).reset_index(drop=True)

        self.data["Date"] = pd.to_datetime(self.data["Date"])

        self.data["DailyReturn"] = (self.data.groupby("Ticker")["Close"].pct_change())

        self.data[f"FutureReturn{self.forecast_horizon}"] = (self.data.groupby("Ticker")["Close"].shift(-self.forecast_horizon)/self.data["Close"]-1)

        spy_close = spy_data["Close"].squeeze()

        spy = (spy_close.rename("SPYClose").reset_index())

        spy["Date"] = pd.to_datetime(spy["Date"])

        spy[f"SPYReturn{self.forecast_horizon}"] = (spy["SPYClose"].shift(-self.forecast_horizon) / spy["SPYClose"] -1)

        self.data = self.data.merge(spy[["Date","SPYClose",f"SPYReturn{self.forecast_horizon}"]],on="Date",how="left")

        self.data[f"NetReturn{self.forecast_horizon}"] = self.data[f"FutureReturn{self.forecast_horizon}"] - self.data[f"SPYReturn{self.forecast_horizon}"]

       # return self.data












