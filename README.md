# AI Finance: Factor-Based Alpha Modeling and Portfolio Optimization

A modular quantitative finance research project for building,
evaluating, and backtesting cross-sectional equity return signals.

The project currently uses historical price and volume data for a
diversified universe of U.S. equities, engineers factor signals,
predicts forward benchmark-relative returns with linear regression, and
converts those predictions into long-short portfolios using both
rank-based and mean-variance portfolio construction.

## Project Pipeline

``` text
DataEngine
    ↓
FactorEngine
    ↓
PreprocessEngine
    ↓
Training
    ↓
Predicted Returns
    ↓
BackTesting
    ├── 20/80 Long-Short Portfolio
    └── Mean-Variance Optimized Portfolio
            ↓
       Ledoit-Wolf Covariance Shrinkage
```

The code is organized into separate modules so that the data pipeline,
factors, prediction model, and portfolio-construction methodology can be
modified independently.

## Current Methodology

### 1. Data Collection

`DataEngine.py` downloads adjusted historical market data using
`yfinance`.

The current configuration uses:

-   Approximately 50 U.S. equities across multiple sectors
-   SPY as the market benchmark
-   Daily close prices and trading volume
-   A configurable forecast horizon
-   Current sample period: 2015-01-01 to 2026-01-01

For each stock, the engine calculates daily returns and forward returns
over the selected forecast horizon.

The benchmark-relative prediction target is

``` text
NetReturnH = FutureReturnH - SPYReturnH
```

where `H` is the configurable forecast horizon. The current experiment
uses `H = 20` trading days.

### 2. Factor Engineering

`FactorEngine.py` constructs the predictive features.

The current factor set consists of:

**Momentum**

``` text
Momentum60_5 = Close[t-5] / Close[t-60] - 1
```

This measures medium-term momentum while skipping the most recent five
trading days.

**Volatility**

``` text
Volatility20 = rolling 20-day standard deviation of daily returns
```

**Relative Volume**

``` text
VolumeRatio20 = Volume / AverageVolume20
LogVolumeRatio20 = log(VolumeRatio20)
```

**Short-Term Return**

``` text
Return5 = Close[t] / Close[t-5] - 1
```

The `FactorEngine` maintains the list of generated features
automatically, making it straightforward to add or remove factors.

### 3. Cross-Sectional Preprocessing

`PreprocessEngine.py` standardizes each factor cross-sectionally on
every trading date:

``` text
z = (x - cross-sectional mean) / cross-sectional standard deviation
```

This expresses each stock's factor value relative to the other stocks
available on the same date.

The data is then split chronologically into:

-   Training: before 2022-01-01
-   Validation: 2022-01-01 through 2023-12-31
-   Test: 2024-01-01 onward

The test period is kept separate from model development and validation.

### 4. Return Prediction

`Training.py` currently implements an OLS linear regression model using
`statsmodels`.

The model predicts future benchmark-relative stock returns from the
standardized factor exposures:

``` text
NetReturnH
    = β0
    + β1 Momentum
    + β2 Volatility
    + β3 LogVolumeRatio
    + β4 ShortTermReturn
    + ε
```

Standard errors are clustered by date to account for cross-sectional
dependence among stocks observed on the same trading day.

Out-of-sample model performance is evaluated on the validation set using
out-of-sample R-squared.

### 5. Portfolio Construction and Backtesting

`BackTesting.py` currently implements two portfolio-construction
approaches.

#### Rank-Based 20/80 Long-Short Portfolio

Stocks are ranked cross-sectionally according to predicted returns at
each rebalance date.

-   Long the top 20%
-   Short the bottom 20%
-   Rebalance every `forecast_horizon` trading days

The resulting long-short return is evaluated using cumulative
performance and annualized Sharpe ratio.

#### Mean-Variance Optimized Portfolio

The second approach uses the model's predicted returns as expected
returns in a mean-variance optimization problem:

``` text
maximize    μᵀw - λ wᵀΣw
```

where:

-   `μ` = predicted forward returns
-   `w` = portfolio weights
-   `Σ` = estimated covariance matrix
-   `λ` = risk-aversion parameter

The current portfolio constraints are:

``` text
sum(w) = 0
||w||₁ <= 2
-0.05 <= w_i <= 0.05
```

These constraints produce a dollar-neutral long-short portfolio, limit
gross exposure to 200%, and restrict each individual stock position to
±5%.

### 6. Covariance Shrinkage

Portfolio risk is estimated using Ledoit-Wolf covariance shrinkage
rather than relying solely on the raw sample covariance matrix.

The general shrinkage idea is

``` text
Σ_shrunk = (1 - δ)S + δF
```

where `S` is the sample covariance matrix, `F` is a more structured
shrinkage target, and `δ` controls the shrinkage intensity.

The shrinkage intensity is estimated from the historical return data
using scikit-learn's `LedoitWolf`.

## Repository Structure

``` text
.
├── main.py
├── DataEngine.py
├── FactorEngine.py
├── PreprocessEngine.py
├── Training.py
├── BackTesting.py
├── tickers.py
└── README.md
```

### Module Responsibilities

  -----------------------------------------------------------------------
  Module                              Responsibility
  ----------------------------------- -----------------------------------
  `DataEngine.py`                     Download and prepare
                                      stock/benchmark data and
                                      forward-return targets

  `FactorEngine.py`                   Construct quantitative factors

  `PreprocessEngine.py`               Cross-sectional normalization and
                                      chronological data splitting

  `Training.py`                       Train the predictive model and
                                      generate validation predictions

  `BackTesting.py`                    Construct portfolios, optimize
                                      weights, and evaluate performance

  `tickers.py`                        Define the equity universe

  `main.py`                           Configure and run the complete
                                      research pipeline
  -----------------------------------------------------------------------

## Installation

Create a Python virtual environment and install the required packages:

``` bash
pip install pandas numpy yfinance statsmodels scikit-learn cvxpy matplotlib
```

## Running the Project

Run:

``` bash
python main.py
```

The current configuration in `main.py` specifies the sample dates,
forecast horizon, factor horizons, covariance lookback, and portfolio
risk-aversion parameter.

For example:

``` python
forecast_horizon = 20
lookback = 60
risk_aversion = 5
```

Changing `forecast_horizon` allows the same framework to be used to
investigate different return-prediction horizons.

## Research Directions

The modular architecture is intended to support controlled experiments
in both alpha modeling and portfolio construction. Planned extensions
include:

-   Comparing different forecast horizons
-   Additional equity factors
-   Ridge and Lasso regression
-   Tree-based and nonlinear machine-learning models
-   Alternative covariance estimators
-   Turnover-aware portfolio optimization
-   Transaction costs
-   Alternative portfolio constraints
-   Risk-aversion and covariance-lookback sensitivity analysis
-   Maximum drawdown and additional performance metrics
-   Final evaluation on the untouched test period

## Research Objective

The broader objective is to study the full quantitative investment
pipeline rather than prediction accuracy in isolation:

``` text
Factor Signals
      ↓
Expected Returns
      ↓
Portfolio Construction
      ↓
Risk Management
      ↓
Out-of-Sample Performance
```

This makes it possible to investigate questions such as whether modest
predictive signals can become economically useful when combined with
appropriate portfolio construction and risk estimation.

## Disclaimer

This repository is a research and educational project. It is not
investment advice, and backtested performance does not imply future
investment performance.
