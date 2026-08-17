import statsmodels.api as sm

class Training:

    def __init__(self, model_type, Train, Validation, Test, standardized_features,FE):

        self.model_type = model_type
        self.Train = Train
        self.Validation = Validation
        self.Test = Test
        self.standardized_features = standardized_features
        self.model = None
        self.FE = FE

    def training(self):

        if self.model_type == "LR":

            x_train = self.Train[self.standardized_features]
            y_train = self.Train[self.FE.target]

            x_train = sm.add_constant(x_train)

            self.model = sm.OLS(y_train,x_train).fit(cov_type="cluster",cov_kwds={"groups": self.Train["Date"]})

        return self.model

    def validation(self):
        if self.model_type == "LR":
	        x_validation = self.Validation[self.standardized_features]
	        
	        x_validation = sm.add_constant(x_validation,has_constant="add")
	
	        self.Validation["PredictedReturn"] = self.model.predict(x_validation)
	
	        actual = self.Validation[self.FE.target]
	
	        predicted = self.Validation["PredictedReturn"]
	
	        oos_r2 = 1-(
	                ((actual-predicted)**2).sum()
	                / (actual**2).sum()
	                )
	
        return self.Validation,oos_r2


