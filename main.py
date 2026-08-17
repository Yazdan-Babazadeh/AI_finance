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



portfolio_results = []


for current_date in rebalance_dates:


    #today is the current_date, what portfolio should i hold for next 20 days

    current_portfolio = portfolio_data[portfolio_data["Date"]==current_date].copy()

    current_tickers = current_portfolio["Ticker"].tolist()

    Mu = current_portfolio["PredictedReturn"].values

    returns_wide = data.pivot(index= "Date", columns = "Ticker", values = "DailyReturn")

    returns_wide = returns_wide.dropna()

    past_returns = returns_wide.loc[returns_wide.index < current_date, current_tickers]

    past_returns = past_returns.tail(lookback)
                           
    # normal method

    Sigma_daily = past_returns.cov().values

    Sigma = 20*Sigma_daily

    #print("Normal Method: ",Sigma)

    #shrinkage method:


    lw = LedoitWolf()
    lw.fit(past_returns)
    Sigma_daily = lw.covariance_
    Sigma = 20 * Sigma_daily

    #print("Shrinkage Method: ", Sigma)

    #print("amount of shrinkage: ", lw.shrinkage_)

    n = len(Mu)

    w = cp.Variable(n)

    expected_portfolio_return = Mu @ w

    portfolio_variance = cp.quad_form(w,Sigma)

    risk_aversion = 5

    objective = cp.Maximize(expected_portfolio_return - risk_aversion*portfolio_variance)

    constraints = [cp.sum(w) ==0, cp.norm1(w) <=2, w<= 0.05, w>=-0.05]

    #constraints = [cp.sum(w) ==1, cp.norm1(w) <=2]

    problem = cp.Problem(objective, constraints)

    problem.solve()

    optimal_weights = w.value

    current_portfolio["Weights"] = optimal_weights


   

    predicted_portfolio_return = (current_portfolio["Weights"]*current_portfolio["PredictedReturn"]).sum()

   

    realized_portfolio_return = (current_portfolio["Weights"]*current_portfolio[FE.target]).sum()

    
    portfolio_results.append({"Date": current_date, "PredictedReturn": predicted_portfolio_return,"RealizedReturn": realized_portfolio_return})


portfolio_results = pd.DataFrame(portfolio_results)

#print(portfolio_results)

portfolio_results["CumulativeRealizedReturn"] = ( 1 + portfolio_results["RealizedReturn"]).cumprod()


portfolio_results["CumulativePredictedReturn"] = ( 1 + portfolio_results["PredictedReturn"]).cumprod()


####### Sharpe Ratio


periods_per_year = 252/20

optimized_sharpe = (portfolio_results["RealizedReturn"].mean()*np.sqrt(12.6))/portfolio_results["RealizedReturn"].std()

#print("Optimized Sharpe Ratio:",optimized_sharpe)
