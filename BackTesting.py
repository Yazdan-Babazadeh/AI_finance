import pandas as pd
import numpy as np


class BackTesting:

    def __init__(self, forecast_horizon, Validation, FE):

        self.forecast_horizon = forecast_horizon

        self.Validation = Validation

        self.FE = FE

    
        self.rebalance_dates = None

        self.portfolio_data = None

    def construct_portfolio(self):

        validation_dates = self.Validation["Date"].drop_duplicates().sort_values()
        self.rebalance_dates = validation_dates.iloc[::self.forecast_horizon]
        self.portfolio_data = self.Validation[self.Validation["Date"].isin(self.rebalance_dates)].copy()

        return self.rebalance_dates,self.portfolio_data

    def backtesting_2080(self):

        self.portfolio_data["Rank"] = (self.portfolio_data.groupby("Date")["PredictedReturn"].rank(pct=True))

        self.portfolio_data["Position"] = 0

        self.portfolio_data.loc[self.portfolio_data["Rank"]>0.8,"Position"] =1
        self.portfolio_data.loc[self.portfolio_data["Rank"]<0.2,"Position"] =-1

        long_returns = self.portfolio_data[self.portfolio_data["Position"]==1].groupby("Date")[self.FE.target].mean()

        short_returns = self.portfolio_data[self.portfolio_data["Position"]==-1].groupby("Date")[self.FE.target].mean()

        backtest = pd.DataFrame({"LongReturn": long_returns, "ShortReturn": short_returns})

        backtest["LongShortReturn"] = backtest["LongReturn"] - backtest["ShortReturn"]

        backtest["CumulativeReturn"] = (1+backtest["LongShortReturn"]).cumprod()

        sharp_ratio = (backtest["LongShortReturn"].mean()*np.sqrt(12.6))/backtest["LongShortReturn"].std()
        
        print("Sharp ratio for 20-80 portfolio equals:", sharp_ratio)
        
        return self.rebalance_dates

