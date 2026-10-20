import os
import sys


from ..Exception import CustomException
from ..Logger import logging

#from source.Configuration import Data_df

#import pandas as pd 
from sklearn.model_selection import train_test_split
from dataclasses import dataclass



@dataclass
class DataIngestionConfig:
    """
    This Class is to save file locations to artifact folder it will save train , test and raw data into it
    """

    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')

class DataIngestion:
    """
    Here we are starting with Data Ingestion and will read dataset making dir structures doing train test splits

    """

    def __init__(self):
        self.ingestion_config =DataIngestionConfig()

    def initate_data_ingestion(self,df,stratify_col=None,test_size=0.2,random_state=14):

        """
        To Initate Data Ingestion use this Function.
        stratify_col: name of a (classification) target column to keep class proportions in the split;
        ignored when some class has fewer than 2 rows.
        """

        logging.info("Entering the Data Ingestion Method or Components")

        try:
            
            #df = Data_df
            logging.info("Read the Data Set as Pandas Dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False, header=True)
            logging.info("Train Test Split will Start, Raw Data Exported")
            stratify = None
            if stratify_col is not None and df[stratify_col].value_counts().min() >= 2:
                stratify = df[stratify_col]
            train_set, test_set = train_test_split(df,test_size=test_size,random_state=random_state,stratify=stratify)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False, header=True) 

            logging.info("Train Test Split is Done , Data in Artifact Folder")
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,     
            )

        except Exception as e:
            raise CustomException(e,sys)
        
DataIngestion.initiate_data_ingestion = DataIngestion.initate_data_ingestion  # correctly spelled alias

# if __name__ == "__main__":
#     obj=DataIngestion()
#     train_data,test_data=obj.initate_data_ingestion()


#     data_transformation =  Data_Transformation()
#     train_arr,test_arr,_ = data_transformation.initiate_data_tranformation(train_path=train_data,test_path=test_data)


#     model_trainer = ModelTrainer()
#     print(model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr))
