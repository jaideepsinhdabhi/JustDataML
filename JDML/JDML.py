#python3

import os
import sys
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore") 


from source.Exception import CustomException
from source.Pipeline import Train_Pipeline
from source.Pipeline.Predict_Pipeline import PredictPipeline,CustomData
from source.Logger import logging

#from source.Configuration import Features_Cols, Data_df, Target_Col, Models, HyperParamter_Yes_or_No, Problem_Objective
from source.Configuration import DataProcessor




class Just_Data_ML:

    """
    Just_Data_ML is a class to train and predict our Values 
    """

    def __init__(self):

        pass

    def Data_Train(self, Features_Cols, Data_df, Target_Col, Models, Problem_Objective, HyperParamter_Yes_or_No = 'No'  , Normalization_tech = 'StandardScaler'):

        """
        Here we are calling Train_pipeline and Training our models and saving it to the files we can use for Prediction.
        """

        try:
            os.makedirs("Output",exist_ok=True)
            train_our_model = Train_Pipeline.TrainingPipeline(
                Features_Cols, Data_df, Target_Col, Models, HyperParamter_Yes_or_No, Problem_Objective, Normalization_tech)
            train_our_model.Train_Model()    

        except Exception as e:
            raise CustomException(e,sys)
        

    def Predict_test(self,Problem_Objective,Features_Cols ,Test_dataDF ):

        """
        Here we are Doing prediction base on our trianed model using in pervious function and importing Predict_pipeline
        it will take Dataframe and as input (you can use csv file and pass it to the function)
        """

        try:
            predict_val = PredictPipeline(Problem_Objective)
            #data = pd.read_csv(self.args.Predict)
            data = Test_dataDF
            input_features = data.columns

            custom_data = CustomData(Features_Cols,**dict.fromkeys(input_features))
            
            for feature in input_features:
                custom_data.__dict__[feature] = data[feature]

            data_frame = custom_data.get_data_as_data_frame()

            final_pred = predict_val.predict(data_frame)
            data["Target_Out"] = final_pred

            return data
            
        except Exception as e:
            raise CustomException(e,sys)
        



def MLtool():
    parser = argparse.ArgumentParser(
        prog="JDML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
            Tool : JustData_ML (JDML)
            Author: Jaideepsinh Dabhi
            emailID: jaideep.dabhi7603@gmail.com
            Usage : JDML.py -C <Data_Config.csv>-T -P <test.csv> -O <Output Folder>
                                         
            * This Tool is Design to Train Machine Learning models and give prediction to help non coder or non tech students to do ML.
            * This Tool will Train models with various paramters and give output.
                        
                *****************\n
                ARUGMENTS  
                -C, --Config    [Compulsory] You need to provide a configuration file to obtain all the necessary arguments for ML models to operate.
                            Make Sure you give Proper Argument mentioned in the Document.
                -T, --Train     [Optional]  If you give this function it will start training your model based on your data_config file.
                            Make Sure you have Data in "Data" Folder.
                -P, --Predict   [Optional]  If you give this function it will do prediction based on your trained model.
                            Make  Sure you don't delete or modify artifact folder to get better results.
                -O, --Output    [Please give with --Predict ]Name of Output csv file (if not given it will store data in Target_outFile csv file)
                            It will store you input test file to output folder with 'Target_Out' Col containing Predictions based on your model.
                                
                        """,
        epilog=" Contact jaideep.dabhi7603@gmail.com for any help or suggestion."
    )

    parser.add_argument('-C', '--Config', type = str, help=" Give a Config file to Application to get details about your task ")
    parser.add_argument('-T', '--Train', action="store_true", default=False, help=" Give a Train argument to Train Data and create a Model")
    parser.add_argument('-P', '--Predict', type=str, help=" Give a Test Data to Predict the Values from the Model")
    parser.add_argument('-O', '--Output', type=str, help=" Give a Output name to Predicted Data Frame from Test Data")

    args, _ = parser.parse_known_args()
    data_processor = DataProcessor(os.path.join("Data", "Data_Config.csv"))
    data_processor = DataProcessor(args.Config)
    data_processor.load_config()
    data_processor.read_data()
    Data_df = data_processor.Data_df
    Features_Cols = data_processor.Features_Cols
    Target_Col = data_processor.Target_Col
    Models = data_processor.Models
    HyperParamter_Yes_or_No = data_processor.HyperParamter_Yes_or_No
    Problem_Objective = data_processor.Problem_Objective
    Normalization_tech = data_processor.scaler

    if not any(vars(args).values()):  # Check if no arguments were provided
        parser.print_usage()
    else:
        jdml_instance = Just_Data_ML()

        if args.Train and not args.Predict:
            jdml_instance.Data_Train(Features_Cols=Features_Cols, Data_df=Data_df, Target_Col=Target_Col, Models=Models,Problem_Objective=Problem_Objective,HyperParamter_Yes_or_No=HyperParamter_Yes_or_No,Normalization_tech=Normalization_tech)
            print("Training of Data is Done please Do prediction using model in artifact folder")


        elif args.Train and args.Predict:
            jdml_instance.Data_Train(Features_Cols=Features_Cols, Data_df=Data_df, Target_Col=Target_Col, Models=Models,Problem_Objective=Problem_Objective,HyperParamter_Yes_or_No=HyperParamter_Yes_or_No,Normalization_tech=Normalization_tech)
            data = pd.read_csv(args.Predict)
            Prediction_final = jdml_instance.Predict_test(Problem_Objective=Problem_Objective,Features_Cols=Features_Cols,Test_dataDF=data)
            if args.Output:
                out_path = os.path.join('Output',f'{args.Output}.csv')
                print(f"Predicted file is saved in {out_path}")
                logging.info(f"Predicted file is saved in {out_path}")
            else:
                out_path = os.path.join('Output','Target_outFile.csv')
                print(f"Predicted file is saved in {out_path} : Note --Output was not given")
                logging.info(f"Predicted file is saved in {out_path} : Note --Output was not given")
                
            print(out_path)
            Prediction_final.to_csv(out_path,index=False)
            print("Training and Prediction Done for given Data")

        elif args.Predict and not args.Train:
            testdata = pd.read_csv(args.Predict)
            Prediction_final = jdml_instance.Predict_test(Problem_Objective=Problem_Objective,Features_Cols=Features_Cols,Test_dataDF=testdata)
            if args.Output:
                out_path = os.path.join('Output',f'{args.Output}.csv')
                print(f"Predicted file is saved in {out_path}")
                logging.info(f"Predicted file is saved in {out_path}")
            else:
                out_path = os.path.join('Output','Target_outFile.csv')
                print(f"Predicted file is saved in {out_path} : Note --Output was not given")
                logging.info(f"Predicted file is saved in {out_path} : Note --Output was not given")
                
            print(out_path)
            Prediction_final.to_csv(out_path,index=False)
            logging.info("Your Model Has Given Prediction Congratulations !!! Your Prediction Files ready in Output Folder")

            print("Model as predicted Values Based on Periously Trained Data in artifact folder")
            
        elif args.Output and not args.Predict:

            print("NO INPUT DATA PLEASE PROVIDE INPUT DATA (-P or --Predict Argument) to Predict Data")

        else:
            print("Some Problem with Arguments")
            parser.print_help()

            
if __name__ == "__main__":
    MLtool()
