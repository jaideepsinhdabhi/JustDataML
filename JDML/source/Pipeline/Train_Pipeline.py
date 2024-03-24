import os
import sys


from source.Logger import logging
from source.Exception import CustomException


from source.components.Data_ingestion import DataIngestion
from source.components.Data_Transformation import Data_Transformation
from source.components.Model_Trainer import ModelTrainer



class TrainingPipeline:

    """
    This is a Training Pipeline to Train the Given Model and 
    Combining all the Fuctions of this tool and do help us to get the best model for prediction
    
    """

    def __init__(self):
        pass

    def Train_Model(self):
        try:
            obj = DataIngestion()
            train_data,test_data = obj.initate_data_ingestion()
            logging.info("Data Splitting is Successful")

            data_transformation =  Data_Transformation()
            train_arr,test_arr,_ = data_transformation.initiate_data_tranformation(train_path=train_data,test_path=test_data)
            logging.info("Data Transformation Done Successfully")

            model_trainer = ModelTrainer()
            logging.info("Data Training is Successful")
            logging.info("Model exported  Successfully")
            print("Best Model Score is : ",model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))
           

        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__ == "__main__":
#     obj = TrainingPipeline()
#     obj.Train_Model()