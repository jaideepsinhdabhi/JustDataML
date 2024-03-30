from dataclasses import dataclass
import sys
import os
import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings("ignore") 


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , StandardScaler, LabelEncoder

from source.Exception import CustomException
from source.Logger import logging
from source.Utils import save_object
#from source.Configuration import Features_Cols, Data_df, Target_Col,Problem_Objective

@dataclass
class Data_transformationConfig:

    """
    A DataClass Config for Data Tranformation it will save locations to save preprocessor pkl file
    """

    preprocessor_obj_file_path = os.path.join('artifact','prerocessor.pkl')
    #DataConfig = ReadConfig("Data_Config.csv")

class Data_Transformation:

    """
    This is a very Import Step in our Program to do Data Tranformation 
    """

#    def __init__(self):
#        self.Data_transformationConfig=Data_transformationConfig()

    def __init__(self, features_cols, data_df, target_col, problem_objective,normalization_tech):
        self.Features_Cols = features_cols
        self.Data_df = data_df
        self.Target_Col = target_col
        self.Problem_Objective = problem_objective
        self.Data_transformationConfig=Data_transformationConfig()
        self.Normalization_tech = normalization_tech



    def get_data_transformer_object(self):
        '''
        This Function is Handling Data Tranformation 
        Here it will Identify Categorical Columns and Numerical Columns and do some processing like
        Imputing the missing values if there are any (Using Simple Imputer)
        Standard Scaling for Numberical Cols and OneHot Encoding for Categorical Cols
        If the Problem is Classification Problem then Target Col (Dependent Col) will be label encoded.

        '''
        try:

            #All_Features = self.Data_transformationConfig.DataConfig["Features_Cols"]
            All_Features = self.Features_Cols
            numerical_columns = self.Data_df[All_Features].select_dtypes(exclude="object").columns   #To get Num Features
            categorical_columns = self.Data_df[All_Features].select_dtypes(include="object").columns #To get Cat Features
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",self.Normalization_tech)
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("one hot encoder",OneHotEncoder())
                    #("scaler",StandardScaler(with_mean=False))
                    #("scaler",self.Normalization_tech(with_mean=False))
                ]
            )
            logging.info("Numerical and Categorical Columns Processed (Scaling and OneHotEncoding Done)")
            logging.info(f"for Numerical Columns we will do scaling with {self.Normalization_tech} )")
            logging.info(f"Categorical columns: {list(categorical_columns)}")
            logging.info(f"Numerical columns: {list(numerical_columns)}")

            preprocessor = ColumnTransformer(
                [
                    ("Num_pipeline",num_pipeline,numerical_columns),
                    ("Cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_tranformation(self,train_path,test_path):

        """
        Here we will do the Data Tranformation using above Pipeline tools. 
        Reading DataFrames (train test both)... Spliting it into Features and Targets... and
         doing Preprocessing before goining into model.
        """

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("The Train and Test Data Imported Sucessfully")

            logging.info('Obtaining Preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = self.Target_Col[0]
            logging.info(f"Target Col is : {target_column_name}")            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            # Do Label Encoding for Target Variable (for Classification Only)
            if self.Problem_Objective=='Classification':
                try:
                    logging.info("Here Problem Objective is  Classification so we have done label encoding for target Variable")
                    label_encoder = LabelEncoder()
                    target_feature_train_df_encoded = label_encoder.fit_transform(target_feature_train_df)
                    target_feature_test_df_encoded = label_encoder.transform(target_feature_test_df)
                    train_arr = np.c_[
                        input_feature_train_arr, np.array(target_feature_train_df_encoded)
                            ]  
                    test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df_encoded)]
                    label_obj_file_path = os.path.join('artifact','Targetlabel.pkl')
                    save_object(label_obj_file_path,label_encoder)
                    logging.info("Here Targetlabel.pkl object is saved to artifact folder")
                except Exception as e:
                    logging.ERROR(e)
                    raise CustomException(e,sys)
            else:
                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.Data_transformationConfig.preprocessor_obj_file_path,
                obj=preprocessor_obj

            )

            return(
                train_arr,
                test_arr,
                self.Data_transformationConfig.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        
