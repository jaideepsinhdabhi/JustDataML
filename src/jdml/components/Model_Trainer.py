import sys

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split


from ..Logger import logging
from ..Options import RunPaths, TrainOptions
from ..Utils import (DEFAULT_METRIC, HIGHER_IS_BETTER, best_model_name as pick_best, evaluate_model,
                     save_object, score_model)
from ..Exception import CustomException

class ModelTrainer:

    """
    Trains and evaluates the selected models for the problem objective (Regression or Classification),
    with hyperparameter tuning when it is set to Yes, and saves the best one.
    """

    def __init__(self, models, hyperparameter_yes_or_no, problem_objective, options=None, paths=None):
        self.Models = models
        self.HyperParamter_Yes_or_No = hyperparameter_yes_or_no
        self.Problem_Objective = problem_objective
        self.options = options or TrainOptions()
        self.paths = paths or RunPaths()
        self.results = {}

    def initiate_model_trainer(self,train_array,test_array):
        """
        Models are compared on a validation split carved out of the training data, so the test set
        is never used for model selection. The winner is refitted on the full training data and
        scored once on the untouched test set; that score is returned. Details for reports and plots
        are kept in self.results.
        """
        try:
            if self.Problem_Objective not in ('Regression', 'Classification'):
                raise ValueError(f"Problem Objective '{self.Problem_Objective}' is not valid; use 'Regression' or 'Classification'.")
            classification = self.Problem_Objective == 'Classification'

            logging.info("Splitting Training and Test input Data into Features and Target")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            if classification:  # labels travel inside a float array; restore the integer class codes
                y_train = y_train.astype(int)
                y_test = y_test.astype(int)

            opts = self.options
            metric = opts.metric or DEFAULT_METRIC[self.Problem_Objective]
            higher = HIGHER_IS_BETTER[metric]
            models = self.Models
            if not models:
                raise ValueError("No models selected for training.")
            logging.info(f"It will use {list(models.keys())} Models for Training ")
            Hyparams= self.HyperParamter_Yes_or_No
            logging.info(f" Hyperparameter tuning setting: [{Hyparams}]")

            stratify = None
            if classification and min(len(y_train[y_train == c]) for c in set(y_train)) >= 2:
                stratify = y_train
            X_fit, X_val, y_fit, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=opts.random_state, stratify=stratify)

            model_report, fitted = evaluate_model(X_train=X_fit,y_train=y_fit,
                                                  X_test=X_val,y_test=y_val,
                                                  models=models,param=Hyparams,prob=self.Problem_Objective,
                                                  eval_name="validation", metric=metric, cv_folds=opts.cv_folds,
                                                  output_dir=self.paths.output_dir, random_state=opts.random_state)

            best_model_name = pick_best(model_report, metric)
            best_model_score = model_report[best_model_name]

            if not higher:
                print(f"The Best model is {best_model_name} with validation {metric} of {best_model_score}")
            elif (best_model_score < 0.8) and (best_model_score >= 0.5):
                logging.info(f'Note : Need Improvement \n The best model is {best_model_name} with validation score {best_model_score}')
            elif best_model_score < 0.5:
                logging.info(f" There is Some ERROR in your model or Code please check Data,Features and Codes \n The best model is {best_model_name} with validation score {best_model_score} ")
            else:
                print(f"The Best model is {best_model_name} with validation score of {best_model_score}")

            best_model = clone(fitted[best_model_name])  # keeps the tuned hyperparameters
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.paths.artifact("model.pkl"),
                obj=best_model
            )

            test_score = score_model(best_model, X_test, y_test, metric)
            logging.info(f"Model is Trained and output is there \n The Best model is {best_model_name} with test {metric} of {test_score}")
            self.results = {
                "best_model_name": best_model_name, "best_model": best_model, "metric": metric,
                "test_score": float(test_score), "validation_scores": {k: float(v) for k, v in model_report.items()},
                "X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
                "y_pred": np.ravel(best_model.predict(X_test)),
            }
            return test_score

        except Exception as e:
            raise CustomException(e,sys)
