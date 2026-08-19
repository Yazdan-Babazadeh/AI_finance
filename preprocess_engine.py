import pandas as pd
import numpy as np

class PreprocessEngine:

    def __init__(self,model_data,factor_engine):

        self.model_data = model_data
        self.factor_engine = factor_engine
        self.standardized_features =[]

    def normalize(self):


        for feature in self.factor_engine.features:

            new_name = feature + "_z"

            self.model_data[new_name] = self.model_data.groupby("Date")[feature].transform(lambda x: (x-x.mean())/x.std())
            self.standardized_features.append(new_name)

            self.model_data = self.model_data.replace([np.inf, -np.inf], np.nan)

            self.model_data = self.model_data.dropna(subset=self.standardized_features + [self.factor_engine.target]).reset_index(drop=True)


    def split_data(self,start_date,validation_date):

        train = self.model_data[self.model_data["Date"]<start_date].copy()
        validation = self.model_data[(self.model_data["Date"] >= start_date) & (self.model_data["Date"] < validation_date)].copy()
        test = self.model_data[self.model_data["Date"] >= validation_date].copy()

        return self.standardized_features ,train, validation, test




        
