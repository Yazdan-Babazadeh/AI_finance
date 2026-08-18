import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from DataEngine import DataEngine
from tickers import tickers
from FactorEngine import FactorEngine
from PreprocessEngine import PreprocessEngine
from Training import Training
from BackTesting import BackTesting
benchmark = "SPY"


# Gathering Data

start_date = "2015-01-01"
end_date = "2026-01-01"
validation_date = "2022-01-01"
test_date = "2024-01-01"
forecast_horizon = 20



DE20 = DataEngine(tickers, benchmark, start_date, end_date, forecast_horizon)

DE20.download_data()

data = DE20.data


#Focusing on 4 factors, Momentum, Volatility, Volume Ratio, Short term Return

### Momentum

FE = FactorEngine(DE20)

FE.add_momentum(60,5)
FE.add_volatility(20)
FE.add_volume(20)
FE.add_shortterm_return(20)

model_data = FE.build_model_data()

PE = PreprocessEngine(model_data,FE)

PE.normalize()

standardized_features, Train, Validation, Test = PE.splitting(validation_date,test_date)


###### Training:

TR = Training("LR",Train,Validation,Test,standardized_features,FE)

model = TR.training()

Validation, oos_r2 = TR.validation()
########## Backtesting:

BT = BackTesting(forecast_horizon,Validation,FE)



rebalance_dates, portfolio_data =BT.construct_portfolio()

BT.backtesting_2080()

######## Portfolio Optimization



lookback = 60
risk_aversion = 5

BT.backtesting_optimized_portfolio(lookback,risk_aversion,data)

