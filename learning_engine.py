import statsmodels.api as sm

class LearningEngine:

    def __init__(self, model_type, train, validation, test, standardized_features,factor_engine):

        self.model_type = model_type
        self.train_data = train
        self.validation_data = validation
        self.test_data = test
        self.standardized_features = standardized_features
        self.model = None
        self.factor_engine = factor_engine

    def train(self):

        if self.model_type == "LR":

            x_train = self.train_data[self.standardized_features]
            y_train = self.train_data[self.factor_engine.target]

            x_train = sm.add_constant(x_train)
            
            

            self.model = sm.OLS(y_train,x_train).fit(cov_type="cluster",cov_kwds={"groups": self.train_data["Date"]})
           # print(self.model.summary())
        return self.model

    def validate(self):
        if self.model_type == "LR":
	        x_validation = self.validation_data[self.standardized_features]
	        
	        x_validation = sm.add_constant(x_validation,has_constant="add")
	
	        self.validation_data["PredictedReturn"] = self.model.predict(x_validation)
	
	        actual = self.validation_data[self.factor_engine.target]
	
	        predicted = self.validation_data["PredictedReturn"]
	
	        oos_r2 = 1-(
	                ((actual-predicted)**2).sum()
	                / (actual**2).sum()
	                )
	
        return self.validation_data,oos_r2


