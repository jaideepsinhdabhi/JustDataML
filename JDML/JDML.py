#python3

import os
import sys
import pandas as pd
import argparse



from source.Exception import CustomException
from source.Pipeline import Train_Pipeline
from source.Pipeline.Predict_Pipeline import PredictPipeline,CustomData
from source.Logger import logging



class Just_Data_ML:

    """
    Just_Data_ML is a class to train and predict our Values 
    """

    def __init__(self,args):
        self.args =args

        
    def Data_Train(self):

        """
        Here we are calling Train_pipeline and Training our models and saving it to the files we can use for Prediction.
        """

        try:
            os.makedirs("Output",exist_ok=True)
            train_ourconfig_model = Train_Pipeline.TrainingPipeline()
            train_ourconfig_model.Train_Model()    

        except Exception as e:
            raise CustomException(e,sys)
        

    def Predict_test(self):

        """
        Here we are Doing prediction base on our trianed model using in pervious function and importing Predict_pipeline
        it will take Dataframe and as input (you can use csv file and pass it to the function)
        """

        try:
            predict_val = PredictPipeline()
            data = pd.read_csv(self.args.Predict)
            input_features = data.columns

            custom_data = CustomData(**dict.fromkeys(input_features))
            
            for feature in input_features:
                custom_data.__dict__[feature] = data[feature]

            data_frame = custom_data.get_data_as_data_frame()

            final_pred = predict_val.predict(data_frame)
            data["Target_Out"] = final_pred
            if self.args.Output:
                out_path = os.path.join('Output',f'{self.args.Output}.csv')
                print(f"Predicted file is saved in {out_path}")
                logging.info(f"Predicted file is saved in {out_path}")
            else:
                out_path = os.path.join('Output',f'Target_outFile.csv')
                print(f"Predicted file is saved in {out_path} : Note --Output was not given")
                logging.info(f"Predicted file is saved in {out_path} : Note --Output was not given")
            data.to_csv(out_path,index=False)
            logging.info("Your Model Has Given Prediction Congratulations !!! Your Prediction Files ready in Output Folder")
        except Exception as e:
            raise CustomException(e,sys)
        



def main():
    parser = argparse.ArgumentParser(
        prog="JDML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
            Tool : JustData_ML (JDML)
            Author: Jaideepsinh Dabhi
            emailID: jaideep.dabhi7603@gmail.com
            Usage : JDML.py -T -P <test.csv> -O <Output Folder>
                                         
            * This Tool is Design to Train Machine Learning models and give prediction to help non coder or non tech students to do ML.
            * This Tool will Train models with various paramters and give output.
                        
                *****************\n
                ARUGMENTS  
                -T, --Train     [Optional]  If you give this function it will start training your model based on your data_config file.
                            Make Sure you have Data_Config_csv file in "Data" Folder.
                -P, --Predict   [Optional]  If you give this function it will do prediction based on your trained model.
                            Make  Sure you don't delete or modify artifact folder to better results.
                -O, --Output    [Please give with --Predict ]Name of Output csv file (if not given it will store data in Target_outFile csv file)
                            It will store you input test file to output folder with 'Target_Out' Col containing Predictions based on your model.
                                
                        """,
        epilog=" Contact jaideep.dabhi7603@gmail.com for any help or suggestion."
    )

    parser.add_argument('-T', '--Train', action="store_true", default=False, help=" Give a Train argument to Train Data and create a Model")
    parser.add_argument('-P', '--Predict', type=str, help=" Give a Test Data to Predict the Values from the Model")
    parser.add_argument('-O', '--Output', type=str, help=" Give a Output name to Predicted Data Frame from Test Data")

    args, _ = parser.parse_known_args()

    if not any(vars(args).values()):  # Check if no arguments were provided
        parser.print_usage()
    else:
        jdml_instance = Just_Data_ML(args)

        if args.Train and not args.Predict:
            jdml_instance.Data_Train()
            print("Training of Data is Done please Do prediction using model in artifact folder")


        elif args.Train and args.Predict:
            jdml_instance.Data_Train()            
            jdml_instance.Predict_test()
            print("Training and Prediction Done for given Data")

        elif args.Predict and not args.Train:
            jdml_instance.Predict_test()
            print("Model as predicted Values Based on Periously Trained Data in artifact folder")
            
        elif args.Output and not args.Predict:

            print("NO INPUT DATA PLEASE PROVIDE INPUT DATA (-P or --Predict Argument) to Predict Data")

        else:
            print("Some Problem with Arguments")
            parser.print_help()
if __name__ == "__main__":
    main()
