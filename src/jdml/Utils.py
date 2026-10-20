import os
import sys
import csv
import hashlib
import joblib
import pandas as pd


from .Exception import CustomException


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler

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
                "AdaBoost Regressor": AdaBoostRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1)
            }


classfication_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Ridge Classification": RidgeClassifier(),
    "GaussianNB": GaussianNB(),
    "KNeighborsClassifier": KNeighborsClassifier(n_jobs=-1),
    "Decision Tree Classifier":DecisionTreeClassifier(),
    "Random Forest Classifier": RandomForestClassifier(n_jobs=-1),
    "Support Vector Classifier": SVC(),
    "AdaBoost Classifier": AdaBoostClassifier(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGBClassifier": XGBClassifier(n_jobs=-1)
            }


# Correctly spelled alias; the old misspelled names stay importable / accepted in configs.
classification_models = classfication_models
LEGACY_MODEL_NAMES = {"Ridge Classifcaiton": "Ridge Classification"}


parameters_regressor= {
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{

                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
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

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },

                "KNeighborsRegressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'p': [1, 2],
                    'metric': ['minkowski']
                }

            }


parameters_classifier = {
                "Logistic Regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                },

                "Ridge Classification": {
                    'alpha': [0.1, 1, 10]
                },

                "GaussianNB": {},

                "KNeighborsClassifier": {
                    'n_neighbors': [3, 5, 7],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },

                "Decision Tree Classifier":{
                    'max_depth': [None, 10, 20, 30],
                },

                "Random Forest Classifier":{
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
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
                    'max_depth': [3, 5, 7]
                },

                "XGBClassifier": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'gamma': [0, 0.1, 0.2],
                }
            }


Normalization_method = {
    "StandardScaler":StandardScaler(),
    "MinMaxScaler":MinMaxScaler(),
    "RobustScaler":RobustScaler(),
    "MaxAbsScaler":MaxAbsScaler()

}


def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        joblib.dump(obj,file_path)
        with open(_hash_path(file_path),"w") as hash_file:
            hash_file.write(_sha256(file_path))

    except Exception as e:
        raise CustomException(e,sys)


def evaluate_model(X_train,y_train,X_test,y_test,models,param,prob,eval_name="test"):
    """
    Fit a clone of every model on (X_train, y_train) and score it on (X_test, y_test).
    The given model objects are never modified. eval_name labels the scored split in
    the metric columns of Output/Model_Summary.csv (e.g. "validation" or "test").
    Returns (report, fitted): the score per model and the fitted clones.
    """
    try:
        if prob not in ("Regression", "Classification"):
            raise ValueError(f"Problem Objective '{prob}' is not valid; use 'Regression' or 'Classification'.")
        if not models:
            raise ValueError("No models selected for training.")

        report = {}
        fitted = {}
        report_frames = []
        summary_frames = []
        for model_name, base_model in models.items():
            model = clone(base_model)
            if param == "Yes":
                para = (parameters_regressor if prob == "Regression" else parameters_classifier)[model_name]
                # Models that already use n_jobs=-1 internally should not also be parallelised by the grid search.
                grid_jobs = 1 if "n_jobs" in model.get_params() else -1
                gs = GridSearchCV(model,para,cv=3,n_jobs=grid_jobs)
                gs.fit(X_train,y_train)
                result_out_df = pd.DataFrame(gs.cv_results_)
                result_out_df["Model"] = model_name
                report_frames.append(result_out_df)
                model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Regression is scored with R2, classification with accuracy.
            if prob == 'Regression':
                test_model_score = r2_score(y_test, y_test_pred)
                model_metrics = pd.DataFrame({
                'Model': [model_name],
                'MAE_train': [mean_absolute_error(y_train, y_train_pred)],
                'MSE_train': [mean_squared_error(y_train, y_train_pred)],
                'R2_train': [r2_score(y_train, y_train_pred)],
                'Explained_Var_train': [explained_variance_score(y_train, y_train_pred)],
                'Med_Abs_Error_train': [median_absolute_error(y_train, y_train_pred)],
                f'MAE_{eval_name}': [mean_absolute_error(y_test, y_test_pred)],
                f'MSE_{eval_name}': [mean_squared_error(y_test, y_test_pred)],
                f'R2_{eval_name}': [test_model_score],
                f'Explained_Var_{eval_name}': [explained_variance_score(y_test, y_test_pred)],
                f'Med_Abs_Error_{eval_name}': [median_absolute_error(y_test, y_test_pred)]
                })
            else:
                weighted = dict(average='weighted', zero_division=0)
                test_model_score = accuracy_score(y_test, y_test_pred)
                model_metrics = pd.DataFrame({
                'Model': [model_name],
                'Accuracy_train': [accuracy_score(y_train, y_train_pred)],
                'Precision_train': [precision_score(y_train, y_train_pred, **weighted)],
                'Recall_train': [recall_score(y_train, y_train_pred, **weighted)],
                'F1_train': [f1_score(y_train, y_train_pred, **weighted)],
                f'Accuracy_{eval_name}': [test_model_score],
                f'Precision_{eval_name}': [precision_score(y_test, y_test_pred, **weighted)],
                f'Recall_{eval_name}': [recall_score(y_test, y_test_pred, **weighted)],
                f'F1_{eval_name}': [f1_score(y_test, y_test_pred, **weighted)]
                })

            summary_frames.append(model_metrics)
            report[model_name] = test_model_score
            fitted[model_name] = model
            print(f" Model : {model_name} ,  {eval_name} Score : {test_model_score}")

        os.makedirs("Output",exist_ok=True)
        pd.concat(summary_frames).to_csv(os.path.join("Output","Model_Summary.csv"),index=False)
        if report_frames:
            pd.concat(report_frames).to_csv(os.path.join("Output","Hyperparameter_stats.csv"), index=False)
        return report, fitted

    except Exception as e:
        raise CustomException(e,sys)


def _sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path,"rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_path(file_path):
    return f"{file_path}.sha256"


def load_object(file_path):
    """
    Load an artifact written by save_object.
    The SHA-256 recorded at save time is verified before deserializing, so a modified or
    swapped file is rejected. The digest sits next to the artifact, so this detects
    tampering and corruption, not an attacker who can rewrite both files: only load
    artifacts from a directory you trust.
    """
    try:
        if not os.path.exists(_hash_path(file_path)):
            raise ValueError(f"No checksum found for '{file_path}'; refusing to load it. Retrain to regenerate it.")
        with open(_hash_path(file_path)) as hash_file:
            expected = hash_file.read().strip()
        if _sha256(file_path) != expected:
            raise ValueError(f"Checksum mismatch for '{file_path}'; the file was modified. Refusing to load it.")
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e,sys)


def safe_join(base_dir, name):
    """
    Join name under base_dir and refuse anything that resolves outside it
    (absolute paths, '..' segments, symlink escapes).
    """
    base = os.path.realpath(base_dir)
    path = os.path.realpath(os.path.join(base, name))
    if os.path.commonpath([base, path]) != base:
        raise ValueError(f"'{name}' resolves outside of '{base_dir}'")
    return os.path.join(base_dir, os.path.relpath(path, base))


def ReadConfig(file_path):
    try:
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file '{file_path}' not found. Please provide it with --Config.")
        Config_Dict = {}
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                key = row[0].strip()
                values = [value.strip() for value in row[1:] if value.strip()]
                Config_Dict[key] = values

        return Config_Dict
    except Exception as e:
        raise CustomException(e,sys)
