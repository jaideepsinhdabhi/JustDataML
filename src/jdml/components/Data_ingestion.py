import os
import sys


from ..Exception import CustomException
from ..Logger import logging


from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    """
    Locations in the artifact folder where the raw, train and test data are saved.
    """

    train_data_path: str = os.path.join('artifact','train.csv')
    test_data_path: str = os.path.join('artifact','test.csv')
    raw_data_path: str = os.path.join('artifact','data.csv')

class DataIngestion:
    """
    Saves the data and splits it into train and test sets.
    """

    def __init__(self):
        self.ingestion_config =DataIngestionConfig()

    def initate_data_ingestion(self,df,stratify_col=None,test_size=0.2,random_state=14):

        """
        Save the raw data and split it into train and test files.
        stratify_col: name of a (classification) target column to keep class proportions in the split;
        ignored when some class has fewer than 2 rows.
        """

        logging.info("Entering the Data Ingestion Method or Components")

        try:

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
