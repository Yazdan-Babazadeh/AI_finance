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
benchmark = "SPY"

# Gathering Data

start_date = "2015-01-01"
end_date = "2026-01-01"
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

#grouped_close = data.groupby("Ticker")["Close"]
#
#data["Momentum60_5"] = grouped_close.shift(5)/grouped_close.shift(60) -1
#
#### Volatility
#
#data["Volatility20"] = data.groupby("Ticker")["DailyReturn"].transform(lambda x: x.rolling(20).std())
#
#### Volume Ratio
#
#data["AvgVolume20"] = data.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
#data["VolumeRatio"] = data["Volume"]/data["AvgVolume20"]
#
#data["LogVolumeRatio"] = data["VolumeRatio"].transform(np.log)
#
#data = data.replace([np.inf,-np.inf],np.nan)
#
#### Short-Term return
#
#
#data["Return5"] = data["Close"]/grouped_close.shift(5) - 1
#
#
########### Defining features and target
#
#
#features = ["Momentum60_5","Volatility20","LogVolumeRatio","Return5"]
#
#target = "NetReturn20"
#
#
#model_data = data[ ["Date","Ticker"] + features + [target]].copy()
#
#model_data = model_data.dropna()
#
#

########## Preproccesing

standardized_features = []
for feature in FE.features:

    new_name = feature + "_z"

    model_data[new_name] = model_data.groupby("Date")[feature].transform(lambda x: (x-x.mean())/x.std())
    standardized_features.append(new_name)


model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna(
    subset=standardized_features + [FE.target]
).reset_index(drop=True)

model_data.to_csv("data.csv")
######### Splitting Data


Train = model_data[model_data["Date"]< "2022-01-01"].copy()
Validation = model_data[(model_data["Date"] >= "2022-01-01") & (model_data["Date"] < "2024-01-01")].copy()

Test = model_data[model_data["Date"] >= "2024-01-01"].copy()

######### Linear Regression

###### Training:

X_train = Train[standardized_features]
y_train = Train[FE.target]

X_train = sm.add_constant(X_train)

model = sm.OLS(
    y_train,
    X_train
).fit(
    cov_type="cluster",
    cov_kwds={
        "groups": Train["Date"]
    }
)


##### Validation:

x_validation = Validation[standardized_features]
y_train = Validation[FE.target]

x_validation = sm.add_constant(
    x_validation,
    has_constant="add"
)

Validation["PredictedReturn"] = model.predict(x_validation)

actual = Validation[FE.target]
predicted = Validation["PredictedReturn"]


oos_r2 = 1 - (
    ((actual - predicted) ** 2).sum()
    / (actual ** 2).sum()
)


########## Backtesting:


validation_dates = Validation["Date"].drop_duplicates().sort_values()
rebalance_dates = validation_dates.iloc[::20]

portfolio_data = Validation[Validation["Date"].isin(rebalance_dates)].copy()

portfolio_data["Rank"] = (portfolio_data.groupby("Date")["PredictedReturn"].rank(pct=True))

portfolio_data["Position"] = 0

portfolio_data.loc[portfolio_data["Rank"]>0.8,"Position"] = 1
portfolio_data.loc[portfolio_data["Rank"]<0.2,"Position"] = -1


long_returns = (portfolio_data[portfolio_data["Position"]==1].groupby("Date")[FE.target].mean())

short_returns = (portfolio_data[portfolio_data["Position"] == -1].groupby("Date")[FE.target].mean())



backtest = pd.DataFrame({"LongReturn": long_returns, "ShortReturn": short_returns})

backtest["LongShortReturn"] = backtest["LongReturn"] - backtest["ShortReturn"]

backtest["CumulativeReturn"] = (1+backtest["LongShortReturn"]).cumprod()

#plt.plot(backtest.index,backtest["CumulativeReturn"])
#plt.show()

sharp_ratio = (backtest["LongShortReturn"].mean()*np.sqrt(12.6))/backtest["LongShortReturn"].std()

print("sharpe ratio for 20-80 portfolio equals:",sharp_ratio)


#### Construction of portfolio optimization


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

    print("Normal Method: ",Sigma)

    #shrinkage method:


    lw = LedoitWolf()
    lw.fit(past_returns)
    Sigma_daily = lw.covariance_
    Sigma = 20 * Sigma_daily

    print("Shrinkage Method: ", Sigma)

    print("amount of shrinkage: ", lw.shrinkage_)

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

print(portfolio_results)

portfolio_results["CumulativeRealizedReturn"] = ( 1 + portfolio_results["RealizedReturn"]).cumprod()


portfolio_results["CumulativePredictedReturn"] = ( 1 + portfolio_results["PredictedReturn"]).cumprod()


####### Sharpe Ratio


periods_per_year = 252/20

optimized_sharpe = (portfolio_results["RealizedReturn"].mean()*np.sqrt(12.6))/portfolio_results["RealizedReturn"].std()

print("Optimized Sharpe Ratio:",optimized_sharpe)
