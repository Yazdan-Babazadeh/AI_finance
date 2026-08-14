import numpy as np

class FactorEngine:

    def __init__(self, DE):

        self.DE = DE
        self.target = f"NetReturn{DE.forecast_horizon}"
        self.features = []
        self.data = DE.data.copy()

    def add_momentum(self,long_horizon,short_horizon):
        
        grouped_close = self.data.groupby("Ticker")["Close"]
        self.data[f"Momentum{long_horizon}_{short_horizon}"] = grouped_close.shift(short_horizon)/grouped_close.shift(long_horizon) - 1
        
        self.features.append(f"Momentum{long_horizon}_{short_horizon}")


    def add_volatility(self,horizon):

        self.data[f"Volatility{horizon}"] = self.data.groupby("Ticker")["DailyReturn"].transform(lambda x: x.rolling(horizon).std())

        self.features.append(f"Volatility{horizon}")

    def add_volume(self,horizon):

        self.data[f"AvgVolume{horizon}"] = self.data.groupby("Ticker")["Volume"].transform(lambda x:x.rolling(horizon).mean())
        self.data[f"VolumeRatio{horizon}"] = self.data["Volume"]/self.data[f"AvgVolume{horizon}"]
        self.data[f"LogVolumeRatio{horizon}"] = self.data[f"VolumeRatio{horizon}"].transform(np.log)
        self.data = self.data.replace([np.inf,-np.inf],np.nan)
        self.features.append(f"LogVolumeRatio{horizon}")

    def add_shortterm_return(self,horizon):
        
        grouped_close = self.data.groupby("Ticker")["Close"]
        self.data[f"Return{horizon}"] = self.data["Close"]/grouped_close.shift(horizon) -1
        self.features.append(f"Return{horizon}")

    def build_model_data(self):
        
        model_data = self.data[["Date","Ticker"]+self.features + [self.target]].copy()
        model_data = model_data.dropna()

        return model_data

