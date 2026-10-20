import argparse
import os
import sys
import warnings

import pandas as pd

from . import templates
from .Configuration import DataProcessor, remember_run
from .Exception import CustomException
from .Logger import logging, setup_logging
from .Options import RunPaths, TrainOptions
from .Pipeline import Train_Pipeline
from .Pipeline.Predict_Pipeline import CustomData, PredictPipeline, load_saved_features
from .Utils import safe_join


class Just_Data_ML:

    """
    Train models and predict values with them.
    """

    def Data_Train(self, Features_Cols, Data_df, Target_Col, Models, Problem_Objective, HyperParamter_Yes_or_No = 'No'  , Normalization_tech = 'StandardScaler', options=None, paths=None):

        """
        Run the training pipeline and save the best model for later predictions.
        Returns a dict with the best model name, the test score and the features used.
        """

        try:
            paths = paths or RunPaths()
            os.makedirs(paths.output_dir,exist_ok=True)
            train_our_model = Train_Pipeline.TrainingPipeline(
                Features_Cols, Data_df, Target_Col, Models, HyperParamter_Yes_or_No, Problem_Objective, Normalization_tech,
                options or TrainOptions(), paths)
            return train_our_model.Train_Model()

        except Exception as e:
            raise CustomException(e,sys)
        

    def Predict_test(self,Problem_Objective,Features_Cols ,Test_dataDF, paths=None):

        """
        Predict with the model trained by Data_Train.
        Takes a DataFrame (for example from pd.read_csv) and returns a copy with a Target_Out column.
        The feature columns saved by training are used; Features_Cols is only the fallback for
        models trained by an older version (it can be None otherwise).
        """

        try:
            paths = paths or RunPaths()
            Features_Cols = load_saved_features(paths) or Features_Cols
            if not Features_Cols:
                raise ValueError("No saved feature list found; train a model first or pass Features_Cols.")
            missing = [col for col in Features_Cols if col not in Test_dataDF.columns]
            if missing:
                raise ValueError(f"Prediction data is missing the feature columns {missing}.")
            predict_val = PredictPipeline(Problem_Objective, paths)
            data = Test_dataDF.copy()  # never modify the caller's DataFrame

            custom_data = CustomData(Features_Cols,**{feature: data[feature] for feature in Features_Cols})
            data_frame = custom_data.get_data_as_data_frame()

            final_pred = predict_val.predict(data_frame)
            data["Target_Out"] = final_pred

            return data
            
        except Exception as e:
            raise CustomException(e,sys)



def _build_parser():
    parser = argparse.ArgumentParser(
        prog="JDML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""\
            Tool : JustDataML (JDML)
            Author: Jaideepsinh Dabhi

            Trains machine learning models and predicts with the best one, without writing code.

            Commands
                JDML init [-o config.yaml] [--type classification|regression]   write a template config
                JDML train   -C <config>                                         train and save the best model
                JDML predict -C <config> <test.csv> [-O <name>]                  predict a CSV file

            The flags below also work on their own (JDML -C <config> -T -P <test.csv> -O <name>).

            Arguments
                -C, --Config    [Required] Config file (.yaml, .yml or .json; .csv is deprecated).
                -T, --Train     [Optional] Train models using the config. The data file must be in the "Data" folder.
                -P, --Predict   [Optional] Predict a CSV file with the trained model. Keep the "artifact" folder.
                -O, --Output    [Optional, with --Predict] Name of the output CSV file in the "Output" folder
                                (default: Target_outFile). It holds your input plus a 'Target_Out' column.
                        """,
        epilog="Contact jaideep.dabhi7603@gmail.com for help or suggestions."
    )
    parser.add_argument('-C', '--Config', type=str, help="Config file describing your task")
    parser.add_argument('-T', '--Train', action="store_true", default=False, help="Train the models and save the best one")
    parser.add_argument('-P', '--Predict', type=str, help="CSV file to predict with the trained model")
    parser.add_argument('-O', '--Output', type=str, help="Name of the predictions file (saved in the Output folder)")
    return parser


def _init_command(argv):
    parser = argparse.ArgumentParser(prog="JDML init", description="Write a template config file.")
    parser.add_argument('-o', '--out', default="config.yaml", help="File to write (default: config.yaml)")
    parser.add_argument('--type', choices=["classification", "regression"], default="classification")
    parser.add_argument('--force', action="store_true", help="Overwrite the file if it exists")
    args = parser.parse_args(argv)
    if os.path.exists(args.out) and not args.force:
        parser.error(f"'{args.out}' already exists; use --force to overwrite it.")
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write(templates.CLASSIFICATION if args.type == "classification" else templates.REGRESSION)
    print(f"Wrote {args.out}. Edit it, then run: JDML train -C {args.out}")


def _parse_subcommand(command, argv):
    parser = argparse.ArgumentParser(prog=f"JDML {command}")
    parser.add_argument('-C', '--Config', required=True, help="Config file (.yaml, .json)")
    if command == "predict":
        parser.add_argument('file', help="CSV file to predict")
        parser.add_argument('-O', '--Output', help="Name of the predictions file (saved in the Output folder)")
    args = parser.parse_args(argv)
    return argparse.Namespace(Config=args.Config, Train=command == "train",
                              Predict=getattr(args, "file", None), Output=getattr(args, "Output", None))


def MLtool(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "init":
        return _init_command(argv[1:])
    parser = _build_parser()
    if argv and argv[0] in ("train", "predict"):
        args = _parse_subcommand(argv[0], argv[1:])
    else:
        args, _ = parser.parse_known_args(argv)
    warnings.filterwarnings("ignore")

    if not any(vars(args).values()):
        parser.print_help()
        return
    if args.Output and not args.Predict:
        parser.error("--Output needs --Predict <test.csv>.")
    if not (args.Train or args.Predict):
        parser.error("Give --Train and/or --Predict.")
    if not args.Config:
        parser.error("--Config <config file> is required.")
    if args.Predict and not os.path.isfile(args.Predict):
        parser.error(f"Prediction file '{args.Predict}' not found.")
    if args.Config.lower().endswith(".csv"):
        print("Note: CSV config files are deprecated. Run `JDML init` for a YAML template.", file=sys.stderr)

    setup_logging()
    data_processor = DataProcessor(args.Config)
    data_processor.load_config()
    data_processor.read_data()
    jdml_instance = Just_Data_ML()

    # Resolve the output path first so a bad name fails before the slow training.
    train_paths = data_processor.paths_for(training=True) if args.Train else None
    out_path = None
    if args.Predict:
        try:
            predict_dir = (train_paths or data_processor.paths_for(training=False)).predictions_dir
            out_path = safe_join(predict_dir, f"{args.Output or 'Target_outFile'}.csv")
        except (ValueError, FileNotFoundError) as e:
            parser.error(f"Invalid --Output: {e}" if isinstance(e, ValueError) else str(e))

    if args.Train:
        jdml_instance.Data_Train(Features_Cols=data_processor.Features_Cols, Data_df=data_processor.Data_df,
                                 Target_Col=data_processor.Target_Col, Models=data_processor.Models,
                                 Problem_Objective=data_processor.Problem_Objective,
                                 HyperParamter_Yes_or_No=data_processor.HyperParamter_Yes_or_No,
                                 Normalization_tech=data_processor.scaler,
                                 options=data_processor.options, paths=train_paths)
        remember_run(train_paths, data_processor.base_paths)
        print(f"Training is done. The model is saved in the {train_paths.artifact_dir} folder.")

    if args.Predict:
        test_data = pd.read_csv(args.Predict)
        Prediction_final = jdml_instance.Predict_test(Problem_Objective=data_processor.Problem_Objective,
                                                      Features_Cols=data_processor.Features_Cols,
                                                      Test_dataDF=test_data,
                                                      paths=train_paths or data_processor.paths_for(training=False))
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        Prediction_final.to_csv(out_path, index=False)
        print(f"Predicted file is saved in {out_path}")
        logging.info(f"Predicted file is saved in {out_path}")


main = MLtool

if __name__ == "__main__":
    MLtool()
