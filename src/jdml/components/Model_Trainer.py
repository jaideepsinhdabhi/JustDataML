import os
import sys
from dataclasses import dataclass

from sklearn.base import clone
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import train_test_split


from ..Logger import logging
from ..Utils import save_object, evaluate_model
from ..Exception import CustomException


@dataclass
class ModelTrainerConfig:

    """
    Where the trained model is saved.
    """

    train_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:

    """
    Trains and evaluates the selected models for the problem objective (Regression or Classification),
    with hyperparameter tuning when it is set to Yes, and saves the best one.
    """

    def __init__(self, models, hyperparameter_yes_or_no, problem_objective):
        self.Models = models
        self.HyperParamter_Yes_or_No = hyperparameter_yes_or_no
        self.Problem_Objective = problem_objective
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        """
        Models are compared on a validation split carved out of the training data, so the test set
        is never used for model selection. The winner is refitted on the full training data and
        scored once on the untouched test set; that score is returned.
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
                X_train, y_train, test_size=0.2, random_state=14, stratify=stratify)

            model_report, fitted = evaluate_model(X_train=X_fit,y_train=y_fit,
                                                  X_test=X_val,y_test=y_val,
                                                  models=models,param=Hyparams,prob=self.Problem_Objective,
                                                  eval_name="validation")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            if (best_model_score < 0.8) and (best_model_score >= 0.5):
                logging.info(f'Note : Need Improvement \n The best model is {best_model_name} with validation score {best_model_score}')
            elif best_model_score < 0.5:
                logging.info(f" There is Some ERROR in your model or Code please check Data,Features and Codes \n The best model is {best_model_name} with validation score {best_model_score} ")
            else:
                print(f"The Best model is {best_model_name} with validation score of {best_model_score}")

            best_model = clone(fitted[best_model_name])  # keeps the tuned hyperparameters
            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            metric = accuracy_score(y_test,predicted) if classification else r2_score(y_test,predicted)
            logging.info(f"Model is Trained and output is there \n The Best model is {best_model_name} with test score of {metric}")
            return metric

        except Exception as e:
            raise CustomException(e,sys)
