import statsmodels.api as sm
from sklearn.linear_model import Lasso

class LearningEngine:

    def __init__(self, model_type, train, validation, test, standardized_features,factor_engine,jump,alpha):

        self.model_type = model_type
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.standardized_features = standardized_features
        self.model = None
        self.factor_engine = factor_engine
        self.jump = jump
        self.alpha = alpha
    def train(self):
 
        x_train = self.train_data[self.standardized_features]
        y_train = self.train_data[self.factor_engine.target]
        
        if self.model_type == "LR":


            x_train = sm.add_constant(x_train)
            
            self.model = sm.OLS(y_train,x_train).fit(cov_type="cluster",cov_kwds={"groups": self.train_data["Date"]})
           # print(self.model.summary())

        elif self.model_type == "LASSO":
            self.model = Lasso(
                    alpha = self.alpha,
                    fit_intercept = True,
                    max_iter = 10000
                    )
            self.model.fit(
                    x_train,
                    y_train
                    )
        return self.model


    def validate(self):
        x_validation = self.validation_data[self.standardized_features]
        if self.model_type == "LR":

            x_validation = sm.add_constant(
                x_validation,
                has_constant="add"
            )
            self.validation_data["PredictedReturn"] = self.model.predict(
                x_validation
            )

        elif self.model_type == "LASSO":
            
            self.validation_data["PredictedReturn"] = self.model.predict(x_validation)

            # Take every jump-th unique date
        validation_dates = (
            self.validation_data["Date"]
            .drop_duplicates()
            .sort_values()
        )
        sampled_dates = validation_dates.iloc[
            ::self.jump
        ]
            # Keep all stocks for those dates
        independent_validation = self.validation_data[
            self.validation_data["Date"].isin(sampled_dates)
        ]
        actual = independent_validation[self.factor_engine.target]
        predicted = independent_validation["PredictedReturn"]
        oos_r2 = 1 - (
            ((actual - predicted) ** 2).sum()
            / (actual ** 2).sum()
        )
        return independent_validation, oos_r2
