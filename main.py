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


forecast_horizon_values = [
    5,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100,
    110,
    120,
    130,
    140,
    150,
    200
]


alpha_values = [
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    5e-3,
    1e-2
]


# Number of different starting offsets
# used for robustness analysis

n_offsets = 10


############################


max_horizon = max(forecast_horizon_values)

full_results = []


for forecast_horizon in forecast_horizon_values:

    print("\n")
    print("=" * 60)
    print("Forecast Horizon:", forecast_horizon)
    print("=" * 60)

    jump = forecast_horizon


    ##################################
    # DATA
    ##################################

    data_engine = DataEngine(
        tickers,
        benchmark,
        start_date,
        end_date,
        forecast_horizon
    )

    data_engine.download_data()

    data = data_engine.data


    ##################################
    # FACTOR LIBRARY
    ##################################

    factor_engine = FactorEngine(
        data_engine
    )


    # Momentum

    factor_engine.add_momentum(
        100,
        5
    )

    factor_engine.add_momentum(
        60,
        5
    )

    factor_engine.add_momentum(
        20,
        5
    )

    factor_engine.add_momentum(
        10,
        5
    )


    # Volatility

    factor_engine.add_volatility(
        100
    )

    factor_engine.add_volatility(
        60
    )

    factor_engine.add_volatility(
        20
    )

    factor_engine.add_volatility(
        10
    )


    # Volume

    factor_engine.add_volume(
        100
    )

    factor_engine.add_volume(
        60
    )

    factor_engine.add_volume(
        20
    )

    factor_engine.add_volume(
        10
    )


    # Short-Term Return

    factor_engine.add_shortterm_return(
        20
    )

    factor_engine.add_shortterm_return(
        10
    )

    factor_engine.add_shortterm_return(
        5
    )


    ##################################
    # MODEL DATA
    ##################################

    model_data = (
        factor_engine
        .build_model_data()
    )


    ##################################
    # PREPROCESSING
    ##################################

    preprocess_engine = PreprocessEngine(
        model_data,
        factor_engine
    )

    preprocess_engine.normalize()


    (
        standardized_features,
        train_data,
        validation_data,
        test_data
    ) = preprocess_engine.split_data(
        validation_date,
        test_date,
        max_horizon
    )


    features = factor_engine.features


    ##################################
    # LASSO
    ##################################

    if learning_model == "LASSO":

        offset_results = []


        ##################################
        # GENERATE OFFSETS
        ##################################

        number_of_offsets = min(
            n_offsets,
            forecast_horizon
        )

        offset_values = np.unique(
            np.linspace(
                0,
                forecast_horizon - 1,
                number_of_offsets,
                dtype=int
            )
        )


        ##################################
        # OFFSET LOOP
        ##################################

        for offset in offset_values:

            alpha_results = []


            ##################################
            # ALPHA LOOP
            ##################################

            for alpha in alpha_values:

                learning_engine = LearningEngine(
                    learning_model,
                    train_data,
                    validation_data.copy(),
                    test_data,
                    standardized_features,
                    factor_engine,
                    jump,
                    alpha,
                    offset
                )


                model = (
                    learning_engine
                    .train()
                )


                (
                    independent_validation,
                    oos_r2
                ) = learning_engine.validate()


                ##################################
                # SELECTED LASSO FACTORS
                ##################################

                selected_factors = []

                for feature, coefficient in zip(
                    features,
                    model.coef_
                ):

                    if abs(coefficient) > 1e-10:

                        selected_factors.append(
                            (
                                feature,
                                coefficient
                            )
                        )


                ##################################
                # STORE ALPHA RESULT
                ##################################

                alpha_results.append(
                    {
                        "Alpha": alpha,

                        "Validation_R2":
                            oos_r2,

                        "Selected_Factors":
                            selected_factors,

                        "N_Selected":
                            len(
                                selected_factors
                            ),

                        "Validation_Rows":
                            len(
                                independent_validation
                            ),

                        "Validation_Dates":
                            independent_validation[
                                "Date"
                            ].nunique()
                    }
                )


            ##################################
            # BEST ALPHA FOR THIS OFFSET
            ##################################

            alpha_results = pd.DataFrame(
                alpha_results
            )


            alpha_results = (
                alpha_results
                .sort_values(
                    "Validation_R2",
                    ascending=False
                )
                .reset_index(
                    drop=True
                )
            )


            best_alpha_result = (
                alpha_results
                .iloc[0]
            )


            offset_results.append(
                {
                    "Offset":
                        offset,

                    "Best_Alpha":
                        best_alpha_result[
                            "Alpha"
                        ],

                    "Validation_R2":
                        best_alpha_result[
                            "Validation_R2"
                        ],

                    "Selected_Factors":
                        best_alpha_result[
                            "Selected_Factors"
                        ],

                    "N_Selected":
                        best_alpha_result[
                            "N_Selected"
                        ],

                    "Validation_Rows":
                        best_alpha_result[
                            "Validation_Rows"
                        ],

                    "Validation_Dates":
                        best_alpha_result[
                            "Validation_Dates"
                        ]
                }
            )


        ##################################
        # OFFSET ROBUSTNESS RESULTS
        ##################################

        offset_results = pd.DataFrame(
            offset_results
        )


        offset_results.to_csv(
            f"offset_results_H{forecast_horizon}.csv",
            index=False
        )


        mean_r2 = (
            offset_results[
                "Validation_R2"
            ]
            .mean()
        )

        median_r2 = (
            offset_results[
                "Validation_R2"
            ]
            .median()
        )

        std_r2 = (
            offset_results[
                "Validation_R2"
            ]
            .std()
        )

        min_r2 = (
            offset_results[
                "Validation_R2"
            ]
            .min()
        )

        max_r2 = (
            offset_results[
                "Validation_R2"
            ]
            .max()
        )


        ##################################
        # STORE HORIZON RESULT
        ##################################

        full_results.append(
            {
                "Forecast_Horizon":
                    forecast_horizon,

                "Mean_R2":
                    mean_r2,

                "Median_R2":
                    median_r2,

                "Std_R2":
                    std_r2,

                "Min_R2":
                    min_r2,

                "Max_R2":
                    max_r2,

                "N_Offsets":
                    len(
                        offset_values
                    ),

                "Train_Rows":
                    len(
                        train_data
                    )
            }
        )


        print(
            "\nOffset results:"
        )

        print(
            offset_results
        )


        print(
            "\nMean R2:",
            mean_r2
        )

        print(
            "Median R2:",
            median_r2
        )

        print(
            "Std R2:",
            std_r2
        )


    ##################################
    # LINEAR REGRESSION
    ##################################

    elif learning_model == "LR":

        results = []


        for n in range(
            1,
            len(features) + 1
        ):

            for subset in combinations(
                features,
                n
            ):

                standardized_subset = [
                    feature + "_z"
                    for feature in subset
                ]


                learning_engine = LearningEngine(
                    learning_model,
                    train_data,
                    validation_data.copy(),
                    test_data,
                    standardized_subset,
                    factor_engine,
                    jump,
                    0,
                    0
                )


                model = (
                    learning_engine
                    .train()
                )


                (
                    independent_validation,
                    oos_r2
                ) = learning_engine.validate()


                results.append(
                    {
                        "Factors":
                            subset,

                        "N_Factors":
                            len(
                                subset
                            ),

                        "Validation_R2":
                            oos_r2
                    }
                )


        results = pd.DataFrame(
            results
        )


        results = (
            results
            .sort_values(
                "Validation_R2",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        best = results.iloc[0]


        full_results.append(
            {
                "Forecast_Horizon":
                    forecast_horizon,

                "Factors":
                    best[
                        "Factors"
                    ],

                "Validation_R2":
                    best[
                        "Validation_R2"
                    ],

                "Train_Rows":
                    len(
                        train_data
                    ),

                "Validation_Rows":
                    len(
                        independent_validation
                    ),

                "Validation_Dates":
                    independent_validation[
                        "Date"
                    ].nunique()
            }
        )


##################################
# FINAL RESULTS
##################################

full_results = pd.DataFrame(
    full_results
)


full_results.to_csv(
    f"forecast_horizon_{learning_model}_offset_robustness.csv",
    index=False
)


print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)

print(
    full_results
)


##################################
# PLOT
##################################

if learning_model == "LASSO":

    plt.scatter(
        full_results[
            "Forecast_Horizon"
        ],
        full_results[
            "Mean_R2"
        ]
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "Mean Validation OOS R²"
    )

    plt.title(
        "Forecast Horizon vs Mean Validation OOS R²"
    )

    plt.show()


elif learning_model == "LR":

    plt.scatter(
        full_results[
            "Forecast_Horizon"
        ],
        full_results[
            "Validation_R2"
        ]
    )

    plt.xlabel(
        "Forecast Horizon"
    )

    plt.ylabel(
        "Validation OOS R²"
    )

    plt.title(
        "Forecast Horizon vs Validation OOS R²"
    )

    plt.show()



########## Backtesting:

#backtest_engine = BacktestEngine(
#    forecast_horizon,
#    validation_data,
#    factor_engine
#)

#rebalance_dates, portfolio_data = (
#    backtest_engine.construct_portfolio()
#)

#backtest_engine.backtest_rank_portfolio()


######### Portfolio Optimization

#lookback = 60
#risk_aversion = 5

#backtest_engine.backtest_optimized_portfolio(
#    lookback,
#    risk_aversion,
#    data
#)
