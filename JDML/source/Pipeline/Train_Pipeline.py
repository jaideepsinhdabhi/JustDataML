import os
import sys
import warnings
warnings.filterwarnings("ignore") 

from source.Logger import logging
from source.Exception import CustomException


from source.components.Data_ingestion import DataIngestion
from source.components.Data_Transformation import Data_Transformation
from source.components.Model_Trainer import ModelTrainer

#from source.Configuration import Features_Cols, Data_df, Target_Col, Models, HyperParamter_Yes_or_No, Problem_Objective

class TrainingPipeline:

    """
    This is a Training Pipeline to Train the Given Model and 
    Combining all the Fuctions of this tool and do help us to get the best model for prediction
    
    """

    def __init__(self, features_cols, data_df, target_col, models, hyperparameter_yes_or_no, problem_objective,normalization_tech):
        self.Features_Cols = features_cols
        self.Data_df = data_df
        self.Target_Col = target_col
        self.Models = models
        self.HyperParamter_Yes_or_No = hyperparameter_yes_or_no
        self.Problem_Objective = problem_objective
        self.Normalization_tech = normalization_tech
    def Train_Model(self):
        try:
            obj = DataIngestion()
            train_data,test_data = obj.initate_data_ingestion(self.Data_df)
            logging.info("Data Splitting is Successful")

            data_transformation =  Data_Transformation(self.Features_Cols,self.Data_df,self.Target_Col,self.Problem_Objective,self.Normalization_tech)
            train_arr,test_arr,_ = data_transformation.initiate_data_tranformation(train_path=train_data,test_path=test_data)
            logging.info("Data Transformation Done Successfully")

            model_trainer = ModelTrainer(self.Models, self.HyperParamter_Yes_or_No, self.Problem_Objective)
            logging.info("Data Training is Successful")
            logging.info("Model exported  Successfully")
            print("Best Model Score is : ",model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))
           

        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__ == "__main__":
#     obj = TrainingPipeline()
#     obj.Train_Model()