import os
import sys
import csv
import json
import pickle
import pandas as pd

from source.Exception import CustomException
#from source.Configuration import Problem_Objective


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import (
    AdaBoostRegressor,AdaBoostClassifier,
    GradientBoostingRegressor,GradientBoostingClassifier,
    RandomForestRegressor,RandomForestClassifier
)
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor, XGBClassifier
#from catboost import CatBoostRegressor, CatBoostClassifier


regressor_models = {
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(n_jobs=-1),
                "XGBRegressor": XGBRegressor(n_jobs=-1),
                "Support Vector Reg" : SVR(),
                "Linear Ridge" : Ridge(),
                "Linear Lasso" : Lasso(),
                "ElasticNet" : ElasticNet(),
                #"CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1)
            }


classfication_models = {
    "Logistic Regression": LogisticRegression(multi_class='ovr',n_jobs=-1),
    "Ridge Classifcaiton": RidgeClassifier(),
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    "Decision Tree Classifier":DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(n_jobs=-1),
    "Support Vector Classifier": SVC(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGBClassifier": XGBClassifier(n_jobs=-1)
            }



parameters_regressor= {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Support Vector Reg":{
                    'C':[0.1,1,10],
                    'kernel': ['linear', 'rbf', 'poly']
                },
                "Linear Ridge":{
                    'alpha': [0.1, 1, 10]
                },
                "Linear Lasso":{
                    'alpha': [0.1, 1, 10]
                },
                "ElasticNet":{
                    'alpha': [0.1, 1, 10],
                    'l1_ratio': [0.1, 0.5, 0.9]
                },

                #"CatBoosting Regressor":{
                #    'depth': [6,8,10],
                #    'learning_rate': [0.01, 0.05, 0.1],
                #    'iterations': [30, 50, 100]
                #},
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    #'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    #'leaf_size': [10, 20, 30, 40, 50],
                    'p': [1, 2],
                    'metric': ['minkowski']
                }

            }


parameters_classifier = {
                "Logistic Regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                },

                "Ridge Classifcaiton": {
                    'alpha': [0.1, 1, 10]
                },

                "GaussianNB": {},

                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7],
                    #'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },

                "Decision Tree Classifier":{
                    'max_depth': [None, 10, 20, 30],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4]
                },

                "Random Forest Classifier":{
                    'n_estimators': [100, 200, 300],
                    #'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth': [None, 10, 20, 30],
                    #'min_samples_split': [2, 5, 10],
                    #'min_samples_leaf': [1, 2, 4],
                    #'bootstrap': [True, False]
                },

                "Support Vector Classifier":{
                    'C': [0.1, 1, 10],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto']
                },

                "AdaBoost Classifier":{
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                },

                "Gradient Boosting Classifier": {
                    'n_estimators': [50, 100, 200],
                    #'learning_rate': [0.01, 0.1, 1.0],
                    'max_depth': [3, 5, 7]
                },

                "XGBClassifier": {
                    'n_estimators': [50, 100, 200],
                    #'learning_rate': [0.01, 0.1, 1.0],
                    'max_depth': [3, 5, 7],
                    'gamma': [0, 0.1, 0.2],
                    #'subsample': [0.8, 1.0],
                    #'colsample_bytree': [0.8, 1.0]
                }
            }







def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        #logging.info(f"{file_obj} is saved")

    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models,param,prob):
    try:
        
        report = {}
        report_df = pd.DataFrame()
        model_summary = pd.DataFrame()
        for model_name, model in models.items():
            #model = list(models.values())[i]
            #print(model_name)
            if prob == "Regression":
                para = parameters_regressor[model_name]
            elif prob == "Classification":
                para = parameters_classifier[model_name]
            #print(para)
            else:
                print("Problem Objective is not Properly Defined")
            if param == "Yes":
                gs = GridSearchCV(model,para,cv=3,n_jobs=-1)
                gs.fit(X_train,y_train)
                result_out = gs.cv_results_
                result_out_df = pd.DataFrame(result_out)
                result_out_df["Model"] = model_name
                report_df = pd.concat([report_df, result_out_df]) 
                report_df.to_csv(os.path.join("Output","Hyperparameter_stats.csv"), index=False)
                model.set_params(**gs.best_params_)
                model.fit(X_train,y_train)
            else:
                model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            #train_model_score = r2_score(y_train,y_train_pred)

            # For Evaluation of our Model for reg we will use R-sq value and for clf we will use accuracy. for testing

            if prob == 'Regression':
                
                mae_train = mean_absolute_error(y_train, y_train_pred)
                mse_train = mean_squared_error(y_train, y_train_pred)
                #rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
                r2_train = r2_score(y_train, y_train_pred)
                explained_var_train = explained_variance_score(y_train, y_train_pred)
                med_abs_error_train = median_absolute_error(y_train, y_train_pred)

                mae_test = mean_absolute_error(y_test, y_test_pred)
                mse_test = mean_squared_error(y_test, y_test_pred)
                #rmse_test = mean_squared_error(y_test, y_test_pred)
                r2_test = r2_score(y_test, y_test_pred)
                explained_var_test = explained_variance_score(y_test, y_test_pred)
                med_abs_error_test = median_absolute_error(y_test, y_test_pred)
                test_model_score = r2_test
            # Construct DataFrame for metrics
                model_metrics = pd.DataFrame({
                'Model': [model_name],
                'MAE_train': [mae_train],
                'MSE_train': [mse_train],
                #'RMSE_train': [rmse_train],
                'R2_train': [r2_train],
                'Explained_Var_train': [explained_var_train],
                'Med_Abs_Error_train': [med_abs_error_train],
                'MAE_test': [mae_test],
                'MSE_test': [mse_test],
                #'RMSE_test': [rmse_test],
                'R2_test': [r2_test],
                'Explained_Var_test': [explained_var_test],
                'Med_Abs_Error_test': [med_abs_error_test]
                })


            elif prob == 'Classification':
                # Compute classification metrics
                accuracy_train = accuracy_score(y_train, y_train_pred)
                precision_train = precision_score(y_train, y_train_pred, average='weighted')
                recall_train = recall_score(y_train, y_train_pred, average='weighted')
                f1_train = f1_score(y_train, y_train_pred, average='weighted')

                accuracy_test = accuracy_score(y_test, y_test_pred)
                precision_test = precision_score(y_test, y_test_pred, average='weighted')
                recall_test = recall_score(y_test, y_test_pred, average='weighted')
                f1_test = f1_score(y_test, y_test_pred, average='weighted')
                test_model_score = accuracy_test

                # Confusion Matrix
                conf_matrix = confusion_matrix(y_test, y_test_pred)


                model_metrics = pd.DataFrame({
                'Model': [model_name],
                'Accuracy_train': [accuracy_train],
                'Precision_train': [precision_train],
                'Recall_train': [recall_train],
                'F1_train': [f1_train],
                'Accuracy_test': [accuracy_test],
                'Precision_test': [precision_test],
                'Recall_test': [recall_test],
                'F1_test': [f1_test]
            })


            else:
                print("Wrong Problem Objective")
            
            model_summary = pd.concat([model_summary, model_metrics])
            model_summary.to_csv(os.path.join("Output","Model_Summary.csv"),index=False)
            report[model_name] = test_model_score
            print(f" Model : {model_name} ,  Test Scores : {test_model_score}")
            
            # print(report)
        return report

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def ReadConfig(file_path):
    try:
        output_json = "".join([file_path.split(".")[0],".json"])
        Config_Dict = {}
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                key = row[0]
                values = [value for value in row[1:] if value]  # Exclude empty values
                Config_Dict[key] = values
        with open(output_json,'w') as json_file:
            json.dump(Config_Dict,json_file,indent=4)

        return Config_Dict
    except Exception as e:
        raise CustomException(e,sys)
    




