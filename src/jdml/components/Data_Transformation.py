from dataclasses import dataclass
import sys
import os
import numpy as np
import pandas as pd


from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder , LabelEncoder

from ..Exception import CustomException
from ..Logger import logging
from ..Utils import save_object

@dataclass
class Data_transformationConfig:

    """
    Where the fitted preprocessor is saved.
    """

    preprocessor_obj_file_path = os.path.join('artifact','preprocessor.pkl')

class Data_Transformation:

    """
    Builds, fits and saves the preprocessing steps for the features and target.
    """


    def __init__(self, features_cols, data_df, target_col, problem_objective,normalization_tech):
        self.Features_Cols = features_cols
        self.Data_df = data_df
        self.Target_Col = target_col
        self.Problem_Objective = problem_objective
        self.Data_transformationConfig=Data_transformationConfig()
        self.Normalization_tech = normalization_tech


    def get_data_transformer_object(self):
        '''
        Build the preprocessor: numeric columns get median imputation and the chosen scaler,
        other columns get most-frequent imputation and one-hot encoding.
        For classification the target is label encoded separately.
        '''
        try:

            All_Features = self.Features_Cols
            numerical_columns = self.Data_df[All_Features].select_dtypes(include="number").columns
            categorical_columns = self.Data_df[All_Features].select_dtypes(exclude="number").columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",self.Normalization_tech)
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ("one hot encoder",OneHotEncoder(handle_unknown="ignore",sparse_output=False))
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
                ],
                sparse_threshold=0
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_tranformation(self,train_path,test_path):

        """
        Read the train and test files, split features from the target,
        fit the preprocessor on the train features and apply it to both.
        """

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The Train and Test Data imported successfully")

            logging.info('Obtaining Preprocessor object')
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = self.Target_Col[0]
            logging.info(f"Target Col is : {target_column_name}")
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            # Do Label Encoding for Target Variable (for Classification Only)
            if self.Problem_Objective=='Classification':
                try:
                    logging.info("Here Problem Objective is  Classification so we have done label encoding for target Variable")
                    label_encoder = LabelEncoder()
                    # Fit on every label so a class missing from the train split cannot break the test encoding.
                    label_encoder.fit(self.Data_df[target_column_name])
                    target_feature_train_df_encoded = label_encoder.transform(target_feature_train_df)
                    target_feature_test_df_encoded = label_encoder.transform(target_feature_test_df)
                    train_arr = np.c_[
                        input_feature_train_arr, np.array(target_feature_train_df_encoded)
                            ]
                    test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df_encoded)]
                    label_obj_file_path = os.path.join('artifact','Targetlabel.pkl')
                    save_object(label_obj_file_path,label_encoder)
                    logging.info("Here Targetlabel.pkl object is saved to artifact folder")
                except Exception as e:
                    logging.error(e)
                    raise CustomException(e,sys)
            else:
                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

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
