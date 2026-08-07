import pandas as pd
import statsmodels.api as sm
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


#Preparing SPY data


spy_close = spy_data["Close"].squeeze()
spy = (spy_close.rename("SPYClose").reset_index())

spy["Date"] = pd.to_datetime(spy["Date"])


spy["SPYReturn20"] = (spy["SPYClose"].shift(-20) / spy["SPYClose"]  - 1)

data = data.merge(spy[["Date","SPYClose","SPYReturn20"]],on="Date",how="left")


data["NetReturn20"] = data["FutureReturn20"] - data["SPYReturn20"]



#Focusing on 4 factors, Momentum, Volatility, Volume Ratio, Short term Return

### Momentum

grouped_close = data.groupby("Ticker")["Close"]

data["Momentum60_5"] = grouped_close.shift(5)/grouped_close.shift(60) -1

### Volatility

data["Volatility20"] = data.groupby("Ticker")["DailyReturn"].transform(lambda x: x.rolling(20).std())

### Volume Ratio

data["AvgVolume20"] = data.groupby("Ticker")["Volume"].transform(lambda x: x.rolling(20).mean())
data["VolumeRatio"] = data["Volume"]/data["AvgVolume20"]

data["LogVolumeRatio"] = data["VolumeRatio"].transform(np.log)

data = data.replace([np.inf,-np.inf],np.nan)

### Short-Term return


data["Return5"] = data["Close"]/grouped_close.shift(5) - 1


########## Defining features and target


features = ["Momentum60_5","Volatility20","LogVolumeRatio","Return5"]

target = "NetReturn20"


model_data = data[ ["Date","Ticker"] + features + [target]].copy()

model_data = model_data.dropna()



########## Preproccesing

standardized_features = []
for feature in features:

    new_name = feature + "_z"

    model_data[new_name] = model_data.groupby("Date")[feature].transform(lambda x: (x-x.mean())/x.std())
    standardized_features.append(new_name)


model_data = model_data.replace(
    [np.inf, -np.inf],
    np.nan
)

model_data = model_data.dropna(
    subset=standardized_features + [target]
).reset_index(drop=True)

model_data.to_csv("data.csv")
######### Splitting Data


Train = model_data[model_data["Date"]< "2022-01-01"].copy()
Validation = model_data[(model_data["Date"] >= "2022-01-01") & (model_data["Date"] < "2024-01-01")].copy()

Test = model_data[model_data["Date"] >= "2024-01-01"].copy()

######### Linear Regression

###### Training:

X_train = Train[standardized_features]
y_train = Train[target]

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
y_train = Validation[target]

x_validation = sm.add_constant(
    x_validation,
    has_constant="add"
)

Validation["PredictedReturn"] = model.predict(x_validation)

actual = Validation[target]
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

print(portfolio_data.head(50))
print(portfolio_data.columns)




