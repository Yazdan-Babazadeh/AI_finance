import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from itertools import combinations

from data_engine import DataEngine
from tickers import tickers
from factor_engine import FactorEngine
from preprocess_engine import PreprocessEngine
from learning_engine import LearningEngine
from backtest_engine import BacktestEngine


##### HYPER PARAMETERS ######

benchmark = "SPY"
learning_model = "LASSO"

start_date = "2010-01-01"
end_date = "2026-01-01"
validation_date = "2020-01-01"
test_date = "2026-01-01"

forecast_horizon_values = [5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150]



alpha = 0.005

############################

max_horizon = max(forecast_horizon_values)
full_results = []
for forecast_horizon in forecast_horizon_values:

    jump = forecast_horizon   
    
    data_engine = DataEngine(tickers, benchmark, start_date, end_date, forecast_horizon)
    
    data_engine.download_data()
    
    data = data_engine.data
    
    
    #Focusing on 4 factors, Momentum, Volatility, Volume Ratio, Short term Return
    
    
    factor_engine = FactorEngine(data_engine)
    
    factor_engine.add_momentum(100,5)
    factor_engine.add_momentum(60,5)
    factor_engine.add_momentum(10,5)
    factor_engine.add_momentum(20,5)
    
    factor_engine.add_volatility(100)
    factor_engine.add_volatility(20)
    factor_engine.add_volatility(10)
    factor_engine.add_volatility(60)

    factor_engine.add_volume(100)
    factor_engine.add_volume(20)
    factor_engine.add_volume(10)
    factor_engine.add_volume(60)

    factor_engine.add_shortterm_return(20)
    factor_engine.add_shortterm_return(10)
    factor_engine.add_shortterm_return(5)
    
    model_data = factor_engine.build_model_data()
    
    
    preprocess_engine = PreprocessEngine(model_data,factor_engine)
    
    preprocess_engine.normalize()
    
    
    standardized_features, train_data, validation_data, test_data = preprocess_engine.split_data(validation_date,test_date,max_horizon)
    
    
    results = []
    features = factor_engine.features
    

    if learning_model == "LASSO":

        learning_engine = LearningEngine(learning_model,train_data,validation_data.copy(),test_data,standardized_features,factor_engine,jump,alpha)
        model = learning_engine.train()

        independent_validation,oos_r2 = learning_engine.validate()

        selected_factors = []

        for feature,coefficient in zip(features,model.coef_):
            if coefficient !=0:

                selected_factors.append((feature,coefficient))

        full_results.append({

            "Forecast_Horizon": forecast_horizon,
            "Alpha": alpha,
            "Selected_Factors": selected_factors,
            "N Selected": len(selected_factors),
            "Validation_R2" : oos_r2,
            "Train_Rows": len(train_data),
            "Validation Rows": len(independent_validation),
            "Validation Dates": (independent_validation["Date"].nunique())
            })



    elif learning_model == "LR":
        
        for n in range(1,len(features)+1):
        
            for subset in combinations(features,n):
        
        
                standardized_subset = [feature +"_z" for feature in subset]
        
        
                learning_engine = LearningEngine(learning_model,train_data,validation_data.copy(),test_data,standardized_subset,factor_engine,jump,alpha)
                model = learning_engine.train()
        
                _, oos_r2 = learning_engine.validate()
        
                results.append({
                    "Factors": subset,
                    "N_Factors": len(subset),
                    "Validation_R2": oos_r2
                    })
        
        results = pd.DataFrame(results)
        results = results.sort_values("Validation_R2",ascending=False).reset_index(drop=True)
        best = results.loc[0]
    
        full_results.append({
            "Forecast_Horizon": forecast_horizon,
            "Factors": best["Factors"],
            "Validation_R2": best["Validation_R2"],
            "Train_Rows": len(train_data),
            "Validation_Rows": len(validation_data),
            "Validation_Dates": validation_data["Date"].nunique()
        })
full_results = pd.DataFrame(full_results)
full_results.to_csv(f"results_jump_{jump}_{learning_model}.csv",index=False)
print(full_results)
    
plt.scatter(full_results["Forecast_Horizon"],full_results["Validation_R2"])
plt.show()
    ########## Backtesting:
    
    #backtest_engine = BacktestEngine(forecast_horizon,validation_data,factor_engine)
    
    #rebalance_dates, portfolio_data =backtest_engine.construct_portfolio()
    
    #backtest_engine.backtest_rank_portfolio()
    
    ######### Portfolio Optimization
    
    #lookback = 60
    #risk_aversion = 5
    
    #backtest_engine.backtest_optimized_portfolio(lookback,risk_aversion,data)
    #
