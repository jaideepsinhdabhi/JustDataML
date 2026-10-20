import os
import sys
import pandas as pd


from ..Logger import logging
from ..Exception import CustomException
from ..Utils import load_object

class PredictPipeline:

    """
    Predicts the target with the model, preprocessor and label encoder saved by training.
    """

    def __init__(self, problem_objective):
        self.Problem_Objective = problem_objective

    def predict(self, features):

        """
        Predict from a DataFrame of feature columns.
        For classification the predictions are decoded back to the original labels.
        """

        try:
            if self.Problem_Objective not in ("Classification", "Regression"):
                raise ValueError(f"Problem Objective '{self.Problem_Objective}' is not valid; use 'Regression' or 'Classification'.")
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join('artifact', 'preprocessor.pkl')

            if self.Problem_Objective=="Classification":
                label_encoder_path = os.path.join('artifact','Targetlabel.pkl')
                label_encoder = load_object(file_path=label_encoder_path)
                logging.info("Target label encoder loaded successfully")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            logging.info("Preprocessor and model loaded successfully")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            if self.Problem_Objective=="Classification":
                preds = preds.astype(int)
                preds = label_encoder.inverse_transform(preds)
                logging.info("Classification done; predictions decoded to the original target values")

            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:

    """
    Holds the feature columns for prediction and converts them into a DataFrame,
    so any dataset can be used without hard-coded column names.
    """


    def __init__(self,features_cols, **kwargs):
        try:
            self.Features_Cols = features_cols
            for feature in self.Features_Cols:
                setattr(self, feature, kwargs.get(feature, None))

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_as_data_frame(self):

        """Return the feature columns as a DataFrame."""

        try:
            custom_data_input_dict = {}
            for feature in self.Features_Cols:
                feature_value = getattr(self,feature)
                custom_data_input_dict[feature] = list(feature_value)
            logging.info("Features loaded successfully")
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
