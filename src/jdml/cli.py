import argparse
import os
import sys
import warnings

import pandas as pd

from .Configuration import DataProcessor
from .Exception import CustomException
from .Logger import logging, setup_logging
from .Pipeline import Train_Pipeline
from .Pipeline.Predict_Pipeline import CustomData, PredictPipeline
from .Utils import safe_join


class Just_Data_ML:

    """
    Train models and predict values with them.
    """

    def Data_Train(self, Features_Cols, Data_df, Target_Col, Models, Problem_Objective, HyperParamter_Yes_or_No = 'No'  , Normalization_tech = 'StandardScaler'):

        """
        Run the training pipeline and save the best model for later predictions.
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
        Predict with the model trained by Data_Train.
        Takes a DataFrame (for example from pd.read_csv) and returns a copy with a Target_Out column.
        """

        try:
            missing = [col for col in Features_Cols if col not in Test_dataDF.columns]
            if missing:
                raise ValueError(f"Prediction data is missing the feature columns {missing}.")
            predict_val = PredictPipeline(Problem_Objective)
            data = Test_dataDF.copy()  # never modify the caller's DataFrame

            custom_data = CustomData(Features_Cols,**{feature: data[feature] for feature in Features_Cols})
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
            Tool : JustDataML (JDML)
            Author: Jaideepsinh Dabhi
            Usage : JDML -C <Data_Config.csv> -T -P <test.csv> -O <Output name>

            Trains machine learning models and predicts with the best one, without writing code.

            Arguments
                -C, --Config    [Required] Configuration file that describes the data and the models to try.
                -T, --Train     [Optional] Train models using the configuration file.
                                The data file must be in the "Data" folder.
                -P, --Predict   [Optional] Predict a CSV file with the trained model.
                                Keep the "artifact" folder created by training.
                -O, --Output    [Optional, with --Predict] Name of the output CSV file in the "Output" folder
                                (default: Target_outFile). It holds your input plus a 'Target_Out' column.
                        """,
        epilog="Contact jaideep.dabhi7603@gmail.com for help or suggestions."
    )

    parser.add_argument('-C', '--Config', type=str, help="Configuration file describing your task")
    parser.add_argument('-T', '--Train', action="store_true", default=False, help="Train the models and save the best one")
    parser.add_argument('-P', '--Predict', type=str, help="CSV file to predict with the trained model")
    parser.add_argument('-O', '--Output', type=str, help="Name of the predictions file (saved in the Output folder)")

    args, _ = parser.parse_known_args()
    warnings.filterwarnings("ignore")

    if not any(vars(args).values()):
        parser.print_help()
        return
    if args.Output and not args.Predict:
        parser.error("--Output needs --Predict <test.csv>.")
    if not (args.Train or args.Predict):
        parser.error("Give --Train and/or --Predict.")
    if not args.Config:
        parser.error("--Config <Data_Config.csv> is required.")
    if args.Predict and not os.path.isfile(args.Predict):
        parser.error(f"Prediction file '{args.Predict}' not found.")

    # Resolve the output path first so a bad name fails before the slow training.
    out_path = None
    if args.Predict:
        try:
            out_path = safe_join('Output', f"{args.Output or 'Target_outFile'}.csv")
        except ValueError as e:
            parser.error(f"Invalid --Output: {e}")

    setup_logging()
    data_processor = DataProcessor(args.Config)
    data_processor.load_config()
    data_processor.read_data()
    jdml_instance = Just_Data_ML()

    if args.Train:
        jdml_instance.Data_Train(Features_Cols=data_processor.Features_Cols, Data_df=data_processor.Data_df,
                                 Target_Col=data_processor.Target_Col, Models=data_processor.Models,
                                 Problem_Objective=data_processor.Problem_Objective,
                                 HyperParamter_Yes_or_No=data_processor.HyperParamter_Yes_or_No,
                                 Normalization_tech=data_processor.scaler)
        print("Training is done. The model is saved in the artifact folder.")

    if args.Predict:
        test_data = pd.read_csv(args.Predict)
        Prediction_final = jdml_instance.Predict_test(Problem_Objective=data_processor.Problem_Objective,
                                                      Features_Cols=data_processor.Features_Cols,
                                                      Test_dataDF=test_data)
        os.makedirs('Output', exist_ok=True)
        Prediction_final.to_csv(out_path, index=False)
        print(f"Predicted file is saved in {out_path}")
        logging.info(f"Predicted file is saved in {out_path}")


main = MLtool

if __name__ == "__main__":
    MLtool()
