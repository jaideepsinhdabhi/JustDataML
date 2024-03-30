import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score, accuracy_score

import warnings
warnings.filterwarnings("ignore") 

from source.Logger import logging
from source.Utils import save_object, evaluate_model
from source.Exception import CustomException
#from source.Configuration import Models, HyperParamter_Yes_or_No,Problem_Objective


@dataclass 
class ModelTrainerConfig:

    """
    Creating a Model Trainer Config to store model inot a pickle file

    """

    train_model_file_path = os.path.join("artifact","model.pkl")

class ModelTrainer:

    """
    Here we are Running the Models and Doing Evalution 
    It will train based on the Problem Objective (Regression or Classification) 
    If you have given Hyperparameter Argument as Yes it will do Hyperparater too for these Models 
    I have tried to include all most frequently used models for both objectives.
    This Function will pick up the best Models and train on that model with best hyper parameter tuning.
    """

    def __init__(self, models, hyperparameter_yes_or_no, problem_objective):
        self.Models = models
        self.HyperParamter_Yes_or_No = hyperparameter_yes_or_no
        self.Problem_Objective = problem_objective
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting Training and Test input Data into Features and Target")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = self.Models
            logging.info(f"It will use {list(models.keys())} Models for Training ")
            Hyparams= self.HyperParamter_Yes_or_No
            logging.info(f" Models are Running with HypterParater Configuration of [{Hyparams}]")

            model_report:dict = evaluate_model(X_train=X_train,y_train=y_train,
                                               X_test=X_test,y_test=y_test,
                                               models= models,param = Hyparams,prob = self.Problem_Objective)

            #print(model_report)
            ## to get the best score from the model
            best_model_score = max(sorted(model_report.values()))
                        
            ## to get the best model name  from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            

            if (best_model_score < 0.8) and (best_model_score >= 0.5):
                #raise CustomException("No Best model Found need improvment",sys)
                logging.info(f'Note : Need Improvement \n The best model is {best_model_name} with score {best_model_score}')
            elif best_model_score < 0.5:
                #raise CustomException("Something is Wrong please check with Data ? Features and codes",sys)
                logging.info(f" There is Some ERROR in your model or Code please check Data,Features and Codes \n The best model is {best_model_name} with score {best_model_score} ")
            else:
                print(f"The Best model is {best_model_name} with score of {best_model_score}")

            logging.info(f"Model is Trained and output is there \n The Best model is {best_model_name} with score of {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            if self.Problem_Objective == 'Regression':
                matric = r2_score(y_test,predicted)
            elif self.Problem_Objective == 'Classification':
                matric = accuracy_score(y_test,predicted)
            else:
                print("Correct Problem Objective")


            #print(f"The best Score is {r2_sq}")
            return matric


        except Exception as e:
            raise CustomException(e,sys)