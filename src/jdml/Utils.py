import os
import sys
import csv
import json
import hashlib
import importlib.util
import warnings
import joblib
import numpy as np
import pandas as pd
import yaml


from .Exception import CustomException


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score, median_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler,MaxAbsScaler

from sklearn.ensemble import (
    AdaBoostRegressor,AdaBoostClassifier,
    GradientBoostingRegressor,GradientBoostingClassifier,
    RandomForestRegressor,RandomForestClassifier,
    HistGradientBoostingRegressor,HistGradientBoostingClassifier
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
                "KNeighborsRegressor": KNeighborsRegressor(n_jobs=-1),
                "HistGradientBoosting Regressor": HistGradientBoostingRegressor()
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
    "XGBClassifier": XGBClassifier(n_jobs=-1),
    "HistGradientBoosting Classifier": HistGradientBoostingClassifier()
            }

# Models of the optional extras (pip install "JustDataML[lightgbm]" / "[catboost]").
# "ALL" in a config means the core models only; name these explicitly to use them.
OPTIONAL_MODEL_EXTRAS = {
    "LightGBM Regressor": "lightgbm", "LightGBM Classifier": "lightgbm",
    "CatBoost Regressor": "catboost", "CatBoost Classifier": "catboost",
}
CORE_REGRESSORS = list(regressor_models)
CORE_CLASSIFIERS = list(classfication_models)

if importlib.util.find_spec("lightgbm"):
    from lightgbm import LGBMClassifier, LGBMRegressor
    regressor_models["LightGBM Regressor"] = LGBMRegressor(n_jobs=-1, verbose=-1)
    classfication_models["LightGBM Classifier"] = LGBMClassifier(n_jobs=-1, verbose=-1)
if importlib.util.find_spec("catboost"):
    from catboost import CatBoostClassifier, CatBoostRegressor
    regressor_models["CatBoost Regressor"] = CatBoostRegressor(verbose=0, allow_writing_files=False)
    classfication_models["CatBoost Classifier"] = CatBoostClassifier(verbose=0, allow_writing_files=False)


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
                },

                "HistGradientBoosting Regressor": {
                    'learning_rate': [.1, .05, .01],
                    'max_iter': [100, 200]
                },

                "LightGBM Regressor": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },

                "CatBoost Regressor": {
                    'depth': [4, 6, 8],
                    'iterations': [100, 300]
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
                },

                "HistGradientBoosting Classifier": {
                    'learning_rate': [.1, .05, .01],
                    'max_iter': [100, 200]
                },

                "LightGBM Classifier": {
                    'learning_rate': [.1, .05, .01],
                    'n_estimators': [50, 100, 200]
                },

                "CatBoost Classifier": {
                    'depth': [4, 6, 8],
                    'iterations': [100, 300]
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


PROBLEM_METRICS = {
    "Classification": ["accuracy", "f1", "precision", "recall", "roc_auc"],
    "Regression": ["r2", "rmse", "mae"],
}
DEFAULT_METRIC = {"Classification": "accuracy", "Regression": "r2"}
HIGHER_IS_BETTER = {"accuracy": True, "f1": True, "precision": True, "recall": True,
                    "roc_auc": True, "r2": True, "rmse": False, "mae": False}
# Scoring names for the hyperparameter search; roc_auc keeps the estimator default.
GRID_SCORING = {"accuracy": "accuracy", "f1": "f1_weighted", "precision": "precision_weighted",
                "recall": "recall_weighted", "r2": "r2", "rmse": "neg_root_mean_squared_error",
                "mae": "neg_mean_absolute_error"}


def _roc_auc(model, X, y):
    scores = model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)
    classes = model.classes_
    if len(classes) == 2:
        return roc_auc_score(y, scores[:, 1] if scores.ndim == 2 else scores)
    return roc_auc_score(label_binarize(y, classes=classes), scores, average="weighted")


def score_model(model, X, y, metric):
    """Score a fitted model with one of the metric names in PROBLEM_METRICS."""
    if metric not in HIGHER_IS_BETTER:
        raise ValueError(f"Unknown metric '{metric}'.")
    if metric == "roc_auc":
        return _roc_auc(model, X, y)
    pred = np.ravel(model.predict(X))
    weighted = dict(average="weighted", zero_division=0)
    if metric == "accuracy":
        return accuracy_score(y, pred)
    if metric == "f1":
        return f1_score(y, pred, **weighted)
    if metric == "precision":
        return precision_score(y, pred, **weighted)
    if metric == "recall":
        return recall_score(y, pred, **weighted)
    if metric == "r2":
        return r2_score(y, pred)
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y, pred)))
    if metric == "mae":
        return mean_absolute_error(y, pred)
    raise ValueError(f"Unknown metric '{metric}'.")


def best_model_name(report, metric):
    """Name of the best model for a metric (the lowest value when lower is better)."""
    pick = max if HIGHER_IS_BETTER[metric] else min
    return pick(report, key=report.get)


def _cross_validate(model, X, y, prob, metric, folds, random_state):
    if prob == "Classification" and np.bincount(y.astype(int)).min() >= folds:
        splitter = StratifiedKFold(folds, shuffle=True, random_state=random_state)
    else:
        splitter = KFold(folds, shuffle=True, random_state=random_state)
    scores = []
    for fit_idx, val_idx in splitter.split(X, y):
        fold_model = clone(model).fit(X[fit_idx], y[fit_idx])
        scores.append(score_model(fold_model, X[val_idx], y[val_idx], metric))
    return scores


def evaluate_model(X_train,y_train,X_test,y_test,models,param,prob,eval_name="test",
                   metric=None,cv_folds=0,output_dir="Output",random_state=14):
    """
    Fit a clone of every model on (X_train, y_train) and score it on (X_test, y_test).
    The given model objects are never modified. eval_name labels the scored split in
    the metric columns of Model_Summary.csv (e.g. "validation" or "test"). metric picks
    the score used to compare models; cv_folds > 1 adds cross-validation columns
    (mean and std on the training part).
    Returns (report, fitted): the score per model and the fitted clones.
    """
    try:
        if prob not in ("Regression", "Classification"):
            raise ValueError(f"Problem Objective '{prob}' is not valid; use 'Regression' or 'Classification'.")
        if not models:
            raise ValueError("No models selected for training.")
        metric = metric or DEFAULT_METRIC[prob]
        if metric not in PROBLEM_METRICS[prob]:
            raise ValueError(f"Metric '{metric}' is not valid for {prob}; use one of {PROBLEM_METRICS[prob]}.")

        report = {}
        fitted = {}
        report_frames = []
        summary_frames = []
        for position, (model_name, base_model) in enumerate(models.items(), 1):
            model = clone(base_model)
            if param == "Yes":
                para = (parameters_regressor if prob == "Regression" else parameters_classifier).get(model_name, {})
                # Models that already use several threads internally should not also be parallelised by the grid search.
                threaded = "n_jobs" in model.get_params() or "thread_count" in model.get_params()
                gs = GridSearchCV(model,para,cv=3,n_jobs=1 if threaded else -1,scoring=GRID_SCORING.get(metric))
                gs.fit(X_train,y_train)
                result_out_df = pd.DataFrame(gs.cv_results_)
                result_out_df["Model"] = model_name
                report_frames.append(result_out_df)
                model.set_params(**gs.best_params_)
            cv_scores = _cross_validate(model, X_train, y_train, prob, metric, cv_folds, random_state) if cv_folds > 1 else None
            model.fit(X_train, y_train)

            y_train_pred = np.ravel(model.predict(X_train))
            y_test_pred = np.ravel(model.predict(X_test))
            test_model_score = score_model(model, X_test, y_test, metric)

            if prob == 'Regression':
                values = {
                'Model': [model_name],
                'MAE_train': [mean_absolute_error(y_train, y_train_pred)],
                'MSE_train': [mean_squared_error(y_train, y_train_pred)],
                'R2_train': [r2_score(y_train, y_train_pred)],
                'Explained_Var_train': [explained_variance_score(y_train, y_train_pred)],
                'Med_Abs_Error_train': [median_absolute_error(y_train, y_train_pred)],
                f'MAE_{eval_name}': [mean_absolute_error(y_test, y_test_pred)],
                f'MSE_{eval_name}': [mean_squared_error(y_test, y_test_pred)],
                f'RMSE_{eval_name}': [float(np.sqrt(mean_squared_error(y_test, y_test_pred)))],
                f'R2_{eval_name}': [r2_score(y_test, y_test_pred)],
                f'Explained_Var_{eval_name}': [explained_variance_score(y_test, y_test_pred)],
                f'Med_Abs_Error_{eval_name}': [median_absolute_error(y_test, y_test_pred)]
                }
            else:
                weighted = dict(average='weighted', zero_division=0)
                values = {
                'Model': [model_name],
                'Accuracy_train': [accuracy_score(y_train, y_train_pred)],
                'Precision_train': [precision_score(y_train, y_train_pred, **weighted)],
                'Recall_train': [recall_score(y_train, y_train_pred, **weighted)],
                'F1_train': [f1_score(y_train, y_train_pred, **weighted)],
                f'Accuracy_{eval_name}': [accuracy_score(y_test, y_test_pred)],
                f'Precision_{eval_name}': [precision_score(y_test, y_test_pred, **weighted)],
                f'Recall_{eval_name}': [recall_score(y_test, y_test_pred, **weighted)],
                f'F1_{eval_name}': [f1_score(y_test, y_test_pred, **weighted)]
                }
            values[f'Score_{eval_name}'] = [test_model_score]
            values['Metric'] = [metric]
            if cv_scores is not None:
                values[f'CV_{metric}_mean'] = [float(np.mean(cv_scores))]
                values[f'CV_{metric}_std'] = [float(np.std(cv_scores))]

            summary_frames.append(pd.DataFrame(values))
            report[model_name] = test_model_score
            fitted[model_name] = model
            print(f" [{position}/{len(models)}] Model : {model_name} ,  {eval_name} {metric} : {test_model_score:.4f}")

        os.makedirs(output_dir,exist_ok=True)
        summary = pd.concat(summary_frames)
        summary.to_csv(os.path.join(output_dir,"Model_Summary.csv"),index=False)
        if report_frames:
            pd.concat(report_frames).to_csv(os.path.join(output_dir,"Hyperparameter_stats.csv"), index=False)
        ranking = summary[["Model", f"Score_{eval_name}"]].sort_values(
            f"Score_{eval_name}", ascending=not HIGHER_IS_BETTER[metric])
        print(ranking.to_string(index=False))
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


# Config keys: canonical name -> accepted spellings (lower case). Old CSV names stay valid.
CONFIG_ALIASES = {
    "Data_Name": ["data_name", "data", "dataset"],
    "Features_Cols": ["features_cols", "features"],
    "Target_Col": ["target_col", "target"],
    "Problem_Objective": ["problem_objective", "problem"],
    "Normalization_tech": ["normalization_tech", "scaler", "normalization"],
    "Model_to_include": ["model_to_include", "models"],
    "HyperParamter_Yes_or_No": ["hyperparamter_yes_or_no", "hyperparameter_yes_or_no", "tuning", "hyperparameter_tuning"],
    "feature_selection": ["feature_selection"],
    "top_k": ["top_k"],
    "correlation_threshold": ["correlation_threshold"],
    "test_size": ["test_size"],
    "random_state": ["random_state", "seed"],
    "metric": ["metric"],
    "cv_folds": ["cv_folds", "cv"],
    "class_weight": ["class_weight"],
    "plots": ["plots"],
    "explain": ["explain"],
    "report": ["report"],
    "datetime_columns": ["datetime_columns", "date_columns"],
    "max_onehot": ["max_onehot", "max_onehot_cardinality"],
    "artifact_dir": ["artifact_dir"],
    "output_dir": ["output_dir"],
    "versioned_runs": ["versioned_runs", "versioned"],
    "run_name": ["run_name"],
}
_ALIAS_LOOKUP = {alias: canon for canon, names in CONFIG_ALIASES.items() for alias in names}


def _as_list(canon, value):
    """Turn a YAML/JSON value into the list of strings the rest of the code works with."""
    if value is None:
        return []
    if isinstance(value, bool):
        if canon == "HyperParamter_Yes_or_No":
            return ["Yes" if value else "No"]
        return ["true" if value else "false"]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, dict):
        raise ValueError(f"'{canon}' must be a value or a list, not a mapping.")
    return [str(value).strip()] if str(value).strip() else []


def config_from_mapping(data, source):
    if not isinstance(data, dict):
        raise ValueError(f"The config file '{source}' must contain a mapping of settings.")
    Config_Dict = {}
    for key, value in data.items():
        if key == "feature_selection" and isinstance(value, dict):
            for sub, sub_value in value.items():
                sub_key = {"method": "feature_selection", "methods": "feature_selection"}.get(sub, sub)
                canon = _ALIAS_LOOKUP.get(str(sub_key).lower())
                if canon not in ("feature_selection", "top_k", "correlation_threshold"):
                    raise ValueError(f"Unknown feature_selection setting '{sub}' in '{source}'.")
                Config_Dict[canon] = _as_list(canon, sub_value)
            continue
        canon = _ALIAS_LOOKUP.get(str(key).strip().lower())
        if canon is None:
            raise ValueError(f"Unknown setting '{key}' in '{source}'. Known settings: {sorted(CONFIG_ALIASES)}")
        Config_Dict[canon] = _as_list(canon, value)
    return Config_Dict


def ReadConfig(file_path):
    """
    Read a .yaml/.yml, .json or (deprecated) .csv config into a dict of canonical key -> list of strings.
    """
    try:
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"Config file '{file_path}' not found. Please provide it with --Config.")
        extension = os.path.splitext(file_path)[1].lower()
        if extension in (".yaml", ".yml"):
            with open(file_path, encoding="utf-8") as file:
                return config_from_mapping(yaml.safe_load(file), file_path)
        if extension == ".json":
            with open(file_path, encoding="utf-8") as file:
                return config_from_mapping(json.load(file), file_path)

        warnings.warn("CSV config files are deprecated; use YAML or JSON (run `JDML init` for a template).",
                      DeprecationWarning, stacklevel=2)
        Config_Dict = {}
        with open(file_path, 'r', encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                key = row[0].strip()
                values = [value.strip() for value in row[1:] if value.strip()]
                Config_Dict[_ALIAS_LOOKUP.get(key.lower(), key)] = values

        return Config_Dict
    except Exception as e:
        raise CustomException(e,sys)
