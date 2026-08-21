import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.covariance import LedoitWolf

class BacktestEngine:

    def __init__(self, forecast_horizon, validation, factor_engine):

        self.forecast_horizon = forecast_horizon

        self.validation = validation

        self.factor_engine = factor_engine

    
        self.rebalance_dates = None

        self.portfolio_data = None
        
        self.periods_per_year = 252/self.forecast_horizon 

    def construct_portfolio(self):

        validation_dates = self.validation["Date"].drop_duplicates().sort_values()
        self.rebalance_dates = validation_dates.iloc[::self.forecast_horizon]
        self.portfolio_data = self.validation[self.validation["Date"].isin(self.rebalance_dates)].copy()

        return self.rebalance_dates,self.portfolio_data

    def backtest_rank_portfolio(self):

        self.portfolio_data["Rank"] = (self.portfolio_data.groupby("Date")["PredictedReturn"].rank(pct=True))

        self.portfolio_data["Position"] = 0

        self.portfolio_data.loc[self.portfolio_data["Rank"]>0.8,"Position"] =1
        self.portfolio_data.loc[self.portfolio_data["Rank"]<0.2,"Position"] =-1

        long_returns = self.portfolio_data[self.portfolio_data["Position"]==1].groupby("Date")[self.factor_engine.target].mean()

        short_returns = self.portfolio_data[self.portfolio_data["Position"]==-1].groupby("Date")[self.factor_engine.target].mean()

        backtest = pd.DataFrame({"LongReturn": long_returns, "ShortReturn": short_returns})

        backtest["LongShortReturn"] = backtest["LongReturn"] - backtest["ShortReturn"]

        backtest["CumulativeReturn"] = (1+backtest["LongShortReturn"]).cumprod()

        sharpe_ratio = (backtest["LongShortReturn"].mean()*np.sqrt(self.periods_per_year))/backtest["LongShortReturn"].std()
        
        print("Sharpe ratio for 20-80 portfolio equals:", sharpe_ratio)
        
        return self.rebalance_dates
    
    def backtest_optimized_portfolio(self,lookback,risk_aversion,data):

        
        
        
        
        portfolio_results = []
        
        
        for current_date in self.rebalance_dates:
        
        
            #today is the current_date, what portfolio should i hold for next 20 days
        
            current_portfolio = self.portfolio_data[self.portfolio_data["Date"]==current_date].copy()
        
            current_tickers = current_portfolio["Ticker"].tolist()
        
            mu = current_portfolio["PredictedReturn"].values
        
            returns_wide = data.pivot(index= "Date", columns = "Ticker", values = "DailyReturn")
        
            returns_wide = returns_wide.dropna()
        
            past_returns = returns_wide.loc[returns_wide.index < current_date, current_tickers]
        
            past_returns = past_returns.tail(lookback)
                                   
            # normal method
        
            sigma_daily = past_returns.cov().values
        
            sigma = self.forecast_horizon *sigma_daily
        
            #print("Normal Method: ",sigma)
        
            #shrinkage method:
        
        
            lw = LedoitWolf()
            lw.fit(past_returns)
            sigma_daily = lw.covariance_
            sigma = self.forecast_horizon * sigma_daily
        
            print("Shrinkage Method: ", sigma)
        
            print("amount of shrinkage: ", lw.shrinkage_)
        
            n = len(mu)
        
            w = cp.Variable(n)
        
            expected_portfolio_return = mu @ w
        
            portfolio_variance = cp.quad_form(w,sigma)
        
        
            objective = cp.Maximize(expected_portfolio_return - risk_aversion*portfolio_variance)
        
        
            constraints = [cp.sum(w) ==0, cp.norm1(w) <=2, w<= 0.05, w>=-0.05]
        
            problem = cp.Problem(objective, constraints)
        
            problem.solve()
        
            optimal_weights = w.value
        
            current_portfolio["Weights"] = optimal_weights
        
        
           
        
            predicted_portfolio_return = (current_portfolio["Weights"]*current_portfolio["PredictedReturn"]).sum()
        
           
        
            realized_portfolio_return = (current_portfolio["Weights"]*current_portfolio[self.factor_engine.target]).sum()
        
            
            portfolio_results.append({"Date": current_date, "PredictedReturn": predicted_portfolio_return,"RealizedReturn": realized_portfolio_return})
        
        
        portfolio_results = pd.DataFrame(portfolio_results)
        
        print(portfolio_results)
        
        portfolio_results["CumulativeRealizedReturn"] = ( 1 + portfolio_results["RealizedReturn"]).cumprod()
        
        
        portfolio_results["CumulativePredictedReturn"] = ( 1 + portfolio_results["PredictedReturn"]).cumprod()
        
        
        ####### Sharpe Ratio
        
        
        
        optimized_sharpe = (portfolio_results["RealizedReturn"].mean()*np.sqrt(self.periods_per_year))/portfolio_results["RealizedReturn"].std()
        
        print("Optimized Sharpe Ratio:",optimized_sharpe)
        
