import sys

from ..Logger import logging
from ..Exception import CustomException
from ..Options import RunPaths, TrainOptions
from .. import Reporting

from ..components.Data_ingestion import DataIngestion
from ..components.Data_Transformation import Data_Transformation
from ..components.Model_Trainer import ModelTrainer


class TrainingPipeline:

    """
    Runs data ingestion, transformation and model training in order and saves the best model.
    """

    def __init__(self, features_cols, data_df, target_col, models, hyperparameter_yes_or_no, problem_objective,normalization_tech,
                 options=None, paths=None):
        self.options = options or TrainOptions()
        self.paths = paths or RunPaths()
        self.Features_Cols = features_cols
        self.Data_df = data_df
        self.Target_Col = target_col
        self.Models = models
        self.HyperParamter_Yes_or_No = hyperparameter_yes_or_no
        self.Problem_Objective = problem_objective
        self.Normalization_tech = normalization_tech
    def Train_Model(self):
        try:
            obj = DataIngestion(self.paths)
            train_data,test_data = obj.initate_data_ingestion(
                self.Data_df,
                stratify_col=self.Target_Col[0] if self.Problem_Objective == 'Classification' else None,
                test_size=self.options.test_size, random_state=self.options.random_state)
            logging.info("Data Splitting is Successful")

            data_transformation =  Data_Transformation(self.Features_Cols,self.Data_df,self.Target_Col,self.Problem_Objective,self.Normalization_tech,
                                                  self.options,self.paths)
            train_arr,test_arr,_ = data_transformation.initiate_data_tranformation(train_path=train_data,test_path=test_data)
            logging.info("Data Transformation Done Successfully")

            model_trainer = ModelTrainer(self.Models, self.HyperParamter_Yes_or_No, self.Problem_Objective,
                                         self.options, self.paths)
            score = model_trainer.initiate_model_trainer(train_array=train_arr,test_array=test_arr)
            logging.info("Data Training is Successful, Model exported Successfully")
            results = model_trainer.results
            print(f"Best Model Test {results['metric']} is : ",score)

            ctx = {
                "results": results, "options": self.options, "paths": self.paths,
                "problem": self.Problem_Objective, "target": self.Target_Col[0], "data_df": self.Data_df,
                "selected_features": data_transformation.selected_features,
                "selection_report": data_transformation.selection_report,
                "feature_names": data_transformation.feature_names,
                "label_encoder": data_transformation.label_encoder,
            }
            Reporting.write_metrics(ctx)
            plots = {}
            if self.options.plots or self.options.report:
                plots = Reporting.make_plots(ctx)
                print(f"Plots are saved in {self.paths.plots_dir}")
            if self.options.report:
                print(f"Report is saved in {Reporting.write_html_report(ctx, plots)}")
            return {
                "best_model_name": results["best_model_name"], "metric": results["metric"],
                "test_score": results["test_score"], "validation_scores": results["validation_scores"],
                "features": data_transformation.selected_features, "plots": plots,
            }


        except Exception as e:
            raise CustomException(e,sys)
