import pandas as pd
import numpy as np

class PreprocessEngine:

    def __init__(self,model_data,FE):

        self.model_data = model_data
        self.FE = FE
        self.standardized_features =[]

    def normalize(self):


        for feature in self.FE.features:

            new_name = feature + "_z"

            self.model_data[new_name] = self.model_data.groupby("Date")[feature].transform(lambda x: (x-x.mean())/x.std())
            self.standardized_features.append(new_name)

            self.model_data = self.model_data.replace([np.inf, -np.inf], np.nan)

            self.model_data = self.model_data.dropna(subset=self.standardized_features + [self.FE.target]).reset_index(drop=True)


    def splitting(self,start_date,validation_date):

        Train = self.model_data[self.model_data["Date"]<start_date].copy()
        Validation = self.model_data[(self.model_data["Date"] >= start_date) & (self.model_data["Date"] < validation_date)].copy()
        Test = self.model_data[self.model_data["Date"] >= validation_date].copy()

        return self.standardized_features ,Train, Validation, Test




        
