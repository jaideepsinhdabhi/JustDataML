import os
import sys
import pandas as pd 

from source.Logger import logging
from source.Exception import CustomException
from source.Utils import load_object
from source.Configuration import Features_Cols, Problem_Objective

class PredictPipeline:

    """"
    Yes if train a model then we also need to use it and predict the values out of it.
    This will predict our Target Variable using Trained best Model and give outcomes.

    """

    def __init__(self):
        pass

    def predict(self, features):

        """
        Given Features in the Predict Funciton it take our test file and output the prediction.
        If classification then it will handle label encoding (from the pkl file) and reverse it to the orignal values.

        """

        try:
            model_path = os.path.join("artifact", "model.pkl")
            preprocessor_path = os.path.join('artifact', 'prerocessor.pkl')
        
            if Problem_Objective=="Classification":
                label_encoder_path = os.path.join('artifact','Targetlabel.pkl')
                label_encoder = load_object(file_path=label_encoder_path)
                logging.info("Here Target_label is also loaded succesfully")
            
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Model Loaded Succesfully")
            logging.info("Preprocessing and model Loaded is Successful")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            if Problem_Objective=="Classification":
                preds = preds.astype(int)
                preds = label_encoder.inverse_transform(preds)
                logging.info("Classification Done Succesfully and decoded the prediction to orignal Target Values")
            
            return preds

        except Exception as e:
            raise CustomException(e, sys)
        
class CustomData:

    """
    Our Program do not need to give Features and DF  hard coded into this,
    It will take care of all the Features .
    CustomData class is designed to dynamically handle features and convert them into a DataFrame,
    providing flexibility in working with different datasets.
    """


    def __init__(self, **kwargs):
        try:

            for feature in Features_Cols:
                # print(feature)
                setattr(self, feature, kwargs.get(feature, None))

        except Exception as e:
            raise CustomException(e, sys) 
    
    def get_data_as_data_frame(self):

        """ Here it will take our Custom data and convert it into Dataframe"""

        try:
            custom_data_input_dict = {}
            for feature in Features_Cols:
                feature_value = getattr(self,feature)
                #print("Fearture: ",feature,"Value: ",type(list(feature_value)))
                custom_data_input_dict[feature] = list(feature_value)
            logging.info("Features Loaded Succesfully")
            #print(custom_data_input_dict)
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



#Finally Prediction going on
# parser = argparse.ArgumentParser(description='Prediction Model for ML Problem ')

# parser.add_argument('-i','-Input', type=str, help='csv File path for Prediction')
# parser.add_argument('-o','-Output',default="Output_predict", type=str, help='Final Output csv Name')
# args = parser.parse_args()
# input_data = args.i
# print("file path for prediction : ",input_data)
# output_name = args.o

# if __name__ == "__main__":
#     obj = PredictPipeline()
#     logging.info("Final Prediction Started")
#     data = pd.read_csv(input_data)
#     logging.info("Data Reading for Prediction Done")
#     input_features = Features_Cols
#     #print('Feature: ',input_features)
#     custom_data = CustomData(**dict.fromkeys(input_features))

#     for feature in input_features:
        
#         custom_data.__dict__[feature] = data[feature]
        
#     data_frame = custom_data.get_data_as_data_frame()
#     #print(data_frame)


#     final_pred = obj.predict(data_frame)
#     #print(final_pred)
    
#     logging.info("Prediction Done Successfully")
    

#     data["Target_Out"] = final_pred

#     out_path = os.path.join('artifact',f'{output_name}.csv')
#     data.to_csv(out_path,index=False)
#     logging.info("Your Model Has Given Prediction Congratulations !!! Your Prediction Files ready in Output Folder")
#     #print(type(final_pred))
    

