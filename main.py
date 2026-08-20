import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf

from data_engine import DataEngine
from tickers import tickers
from factor_engine import FactorEngine
from preprocess_engine import PreprocessEngine
from learning_engine import LearningEngine
from backtest_engine import BacktestEngine

benchmark = "SPY"


# Gathering Data

start_date = "2015-01-01"
end_date = "2026-01-01"
validation_date = "2022-01-01"
test_date = "2024-01-01"
forecast_horizon = 20



data_engine = DataEngine(tickers, benchmark, start_date, end_date, forecast_horizon)

data_engine.download_data()

data = data_engine.data


#Focusing on 4 factors, Momentum, Volatility, Volume Ratio, Short term Return

### Momentum

factor_engine = FactorEngine(data_engine)

factor_engine.add_momentum(60,5)
factor_engine.add_volatility(20)
factor_engine.add_volume(20)
#factor_engine.add_shortterm_return(5)

model_data = factor_engine.build_model_data()

preprocess_engine = PreprocessEngine(model_data,factor_engine)

preprocess_engine.normalize()

standardized_features, train_data, validation_data, test_data = preprocess_engine.split_data(validation_date,test_date)


###### Training:

learning_engine = LearningEngine("LR",train_data,validation_data,test_data,standardized_features,factor_engine)

model = learning_engine.train()

validation_data, oos_r2 = learning_engine.validate()

print(oos_r2)
########## Backtesting:

#backtest_engine = BacktestEngine(forecast_horizon,validation_data,factor_engine)

#rebalance_dates, portfolio_data =backtest_engine.construct_portfolio()

#backtest_engine.backtest_rank_portfolio()

######### Portfolio Optimization

#lookback = 60
#risk_aversion = 5

#backtest_engine.backtest_optimized_portfolio(lookback,risk_aversion,data)
#
