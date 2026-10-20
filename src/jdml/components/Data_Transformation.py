import json
import os
import sys
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, TargetEncoder

from ..Exception import CustomException
from ..Logger import logging
from ..Options import RunPaths, TrainOptions
from ..Utils import save_object
from .Feature_Selection import select_features


class DateTimeFeatures(BaseEstimator, TransformerMixin):
    """Turns date columns (dates or date strings) into year, month, day, weekday and day-of-year numbers."""

    PARTS = ("year", "month", "day", "dayofweek", "dayofyear")

    def fit(self, X, y=None):
        self.columns_ = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        return self

    def transform(self, X):
        frame = pd.DataFrame(X, columns=self.columns_)
        parts = []
        for col in self.columns_:
            dates = pd.to_datetime(frame[col], errors="coerce", format="mixed")
            parts.extend(getattr(dates.dt, part).astype(float) for part in self.PARTS)
        return np.column_stack(parts)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{col}_{part}" for col in self.columns_ for part in self.PARTS])


def parse_dates(frame, columns):
    """Convert the listed columns to datetimes (unparseable values become missing)."""
    frame = frame.copy()
    for col in columns:
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], errors="coerce", format="mixed")
    return frame


class Data_Transformation:

    """
    Selects features, builds, fits and saves the preprocessing steps for the features and target.
    """

    def __init__(self, features_cols, data_df, target_col, problem_objective, normalization_tech,
                 options=None, paths=None):
        self.Features_Cols = features_cols
        self.Data_df = data_df
        self.Target_Col = target_col
        self.Problem_Objective = problem_objective
        self.Normalization_tech = normalization_tech
        self.options = options or TrainOptions()
        self.paths = paths or RunPaths()
        self.label_encoder = None
        self.feature_names = []
        self.selected_features = list(features_cols)
        self.selection_report = None

    def get_data_transformer_object(self, train_features=None, y=None):
        '''
        Build the preprocessor: numeric columns get median imputation and the chosen scaler, date
        columns become numeric parts, text columns with few values get one-hot encoding and
        high-cardinality ones get target encoding.
        '''
        try:
            frame = self.Data_df[self.selected_features] if train_features is None else train_features
            dates = [c for c in frame.columns if c in self.options.datetime_columns
                     or pd.api.types.is_datetime64_any_dtype(frame[c])]
            numerical_columns = [c for c in frame.select_dtypes(include="number").columns if c not in dates]
            categorical_columns = [c for c in frame.columns if c not in dates and c not in numerical_columns]

            can_target_encode = y is not None and not (
                self.Problem_Objective == "Classification" and np.bincount(y.astype(int)).min() < 2)
            low = [c for c in categorical_columns
                   if not can_target_encode or frame[c].nunique() <= self.options.max_onehot]
            high = [c for c in categorical_columns if c not in low]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", clone(self.Normalization_tech))])
            date_pipeline = Pipeline(steps=[
                ("dates", DateTimeFeatures()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", clone(self.Normalization_tech))])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one hot encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
            folds = 5
            if self.Problem_Objective == "Classification" and y is not None:
                folds = max(2, min(5, int(np.bincount(y.astype(int)).min())))
            high_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target encoder", TargetEncoder(cv=folds, random_state=self.options.random_state))])
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Date columns: {dates}")
            logging.info(f"One-hot columns: {low}; target encoded columns: {high}")

            return ColumnTransformer(
                [
                    ("Num_pipeline", num_pipeline, numerical_columns),
                    ("Date_pipeline", date_pipeline, dates),
                    ("Cat_pipeline", cat_pipeline, low),
                    ("HighCard_pipeline", high_pipeline, high),
                ],
                sparse_threshold=0,
                verbose_feature_names_out=False,
            )

        except Exception as e:
            raise CustomException(e,sys)

    def _select(self, train_features, y):
        opts = self.options
        methods = list(opts.feature_selection) or (["importance"] if opts.auto_features else [])
        self.selected_features, self.selection_report = select_features(
            train_features, y, self.Problem_Objective, methods, top_k=opts.top_k,
            correlation_threshold=opts.correlation_threshold, random_state=opts.random_state,
            auto=opts.auto_features)
        self.selection_report.to_csv(self.paths.output("selected_features.csv"), index=False)
        dropped = self.selection_report[~self.selection_report["kept"]]
        for _, row in dropped.iterrows():
            print(f" Dropped feature '{row['feature']}': {row['reason']}")
        with open(self.paths.artifact("features.json"), "w", encoding="utf-8") as handle:
            json.dump({"selected": self.selected_features, "candidates": list(train_features.columns)}, handle, indent=2)

    def initiate_data_tranformation(self,train_path,test_path):

        """
        Read the train and test files, choose the features from the train rows, split features
        from the target, fit the preprocessor on the train features and apply it to both.
        """

        try:
            dates = self.options.datetime_columns
            train_df = parse_dates(pd.read_csv(train_path), dates)
            test_df = parse_dates(pd.read_csv(test_path), dates)
            os.makedirs(self.paths.output_dir, exist_ok=True)

            logging.info("The Train and Test Data imported successfully")

            target_column_name = self.Target_Col[0]
            logging.info(f"Target Col is : {target_column_name}")
            candidates = list(self.Features_Cols)
            input_feature_train_df = train_df[candidates]
            input_feature_test_df = test_df[candidates]
            target_feature_train_df = train_df[target_column_name]
            target_feature_test_df = test_df[target_column_name]

            if self.Problem_Objective == 'Classification':
                logging.info("Classification: label encoding the target")
                self.label_encoder = LabelEncoder()
                # Fit on every label so a class missing from the train split cannot break the test encoding.
                self.label_encoder.fit(self.Data_df[target_column_name])
                y_train = self.label_encoder.transform(target_feature_train_df)
                y_test = self.label_encoder.transform(target_feature_test_df)
                save_object(self.paths.artifact('Targetlabel.pkl'), self.label_encoder)
            else:
                y_train = np.array(target_feature_train_df)
                y_test = np.array(target_feature_test_df)

            self._select(input_feature_train_df, y_train)
            input_feature_train_df = input_feature_train_df[self.selected_features]
            input_feature_test_df = input_feature_test_df[self.selected_features]

            preprocessor_obj = self.get_data_transformer_object(input_feature_train_df, y_train)
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df, y_train)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            self.feature_names = list(preprocessor_obj.get_feature_names_out())

            train_arr = np.c_[input_feature_train_arr, y_train]
            test_arr = np.c_[input_feature_test_arr, y_test]

            save_object(
                file_path=self.paths.artifact('preprocessor.pkl'),
                obj=preprocessor_obj
            )
            logging.info("Saved preprocessing object.")

            return train_arr, test_arr, self.paths.artifact('preprocessor.pkl')

        except Exception as e:
            raise CustomException(e,sys)
