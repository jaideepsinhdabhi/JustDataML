import pandas as pd 
import os
import sys
from source.Utils import ReadConfig , regressor_models, classfication_models
from source.Exception import CustomException




class DataProcessor:
    def __init__(self, config_file):
        self.config_file = config_file
        self.DataConfig = None
        self.Data_df = None
        self.Features_Cols = None
        self.Target_Col = None
        self.Normalization_tech = None
        self.Model_to_include = None
        self.Problem_Objective = None
        self.Models = None
        self.HyperParamter_Yes_or_No = None

    def load_config(self):
        self.DataConfig = ReadConfig(self.config_file)

    def read_data(self):
        try:
            self.Data_df = pd.read_csv(os.path.join("Data", self.DataConfig['Data_Name'][0]))
            self.Features_Cols = self.DataConfig["Features_Cols"]
            self.Target_Col = self.DataConfig["Target_Col"]
            self.Normalization_tech = self.DataConfig["Normalization_tech"]
            self.Model_to_include = self.DataConfig["Model_to_include"]
            self.Problem_Objective = self.DataConfig["Problem_Objective"][0]
            if self.Problem_Objective == "Regression":
                if self.Model_to_include[0] == "ALL":
                    self.Models = regressor_models
                else:
                    self.Models = {key: value for key, value in regressor_models.items() if key in self.Model_to_include}
            elif self.Problem_Objective == "Classification":
                if self.Model_to_include[0] == "ALL":
                    self.Models = classfication_models
                else:
                    self.Models = {key: value for key, value in classfication_models.items() if key in self.Model_to_include}
            else:
                print(f"Problem obj is '{self.Problem_Objective}' this is not properly define please add Proper problem objective i.e 'Regression' or 'Classification'")
            self.HyperParamter_Yes_or_No = self.DataConfig["HyperParamter_Yes_or_No"][0]
        except Exception as e:
            raise CustomException(e,sys)



#Example usage:
ConfigFile = os.path.join("Data","Data_Config.csv")
data_processor = DataProcessor(ConfigFile)
data_processor.load_config()
data_processor.read_data()
Data_df = data_processor.Data_df
Features_Cols = data_processor.Features_Cols
Target_Col = data_processor.Target_Col
Normalization_tech = data_processor.Normalization_tech
Problem_Objective = data_processor.Problem_Objective
Models = data_processor.Models
HyperParamter_Yes_or_No = data_processor.HyperParamter_Yes_or_No


