"""
Python API for notebooks and scripts:

    import jdml
    model = jdml.fit(df, target="price")
    model.report()
    model.predict(new_df)
"""
import pandas as pd

from .Configuration import DataProcessor, remember_run
from .Options import RunPaths
from .Pipeline.Predict_Pipeline import load_saved_features
from .cli import Just_Data_ML

CLASSIFICATION_MAX_UNIQUE = 15


def infer_problem(target):
    """Classification for text, boolean or low-count whole-number targets, otherwise Regression."""
    series = pd.Series(target).dropna()
    if not pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return "Classification"
    whole = (series == series.round()).all()
    return "Classification" if whole and series.nunique() <= CLASSIFICATION_MAX_UNIQUE else "Regression"


class Model:
    """A trained JDML model: .predict(), .report(), .features, .score and friends."""

    def __init__(self, problem, paths, result, features):
        self.problem = problem
        self.paths = paths
        self.features = list(features)
        self.best_model_name = result["best_model_name"]
        self.metric = result["metric"]
        self.score = result["test_score"]
        self.validation_scores = result["validation_scores"]
        self.plots = result["plots"]

    def predict(self, df):
        """Predictions for the rows of df (needs the feature columns). Returns a Series with df's index."""
        out = Just_Data_ML().Predict_test(self.problem, self.features, df, paths=self.paths)
        return out["Target_Out"]

    def report(self):
        """The Model_Summary table: one row per model tried."""
        return pd.read_csv(self.paths.output("Model_Summary.csv"))

    @classmethod
    def load(cls, problem, artifact_dir="artifact", output_dir="Output"):
        """Reopen a model trained earlier (the artifact folder written by fit or the CLI)."""
        paths = RunPaths(artifact_dir, output_dir)
        features = load_saved_features(paths)
        if features is None:
            raise FileNotFoundError(f"No trained model found in '{artifact_dir}'.")
        stub = {"best_model_name": "", "metric": "", "test_score": float("nan"), "validation_scores": {}, "plots": {}}
        return cls(problem, paths, stub, features)

    def __repr__(self):
        return f"Model({self.problem}, best={self.best_model_name!r}, {self.metric}={self.score:.4f}, features={len(self.features)})"


def fit(df, target, problem=None, features="auto", models="ALL", scaler="StandardScaler", tuning=False,
        metric=None, feature_selection=None, top_k=0, cv_folds=0, class_weight=None, plots=False,
        report=False, explain="none", test_size=0.2, random_state=14, datetime_columns=None,
        artifact_dir="artifact", output_dir="Output"):
    """
    Train models on a DataFrame and keep the best one. Settings match the YAML config
    (see the README); problem=None guesses Classification or Regression from the target.
    Files are written to artifact_dir and output_dir like the CLI does.
    """
    problem = problem or infer_problem(df[target])
    config = {
        "target": target, "problem": problem, "features": features, "models": models, "scaler": scaler,
        "tuning": tuning, "top_k": top_k, "cv_folds": cv_folds, "plots": plots, "report": report,
        "explain": explain, "test_size": test_size, "random_state": random_state,
        "artifact_dir": artifact_dir, "output_dir": output_dir,
    }
    for key, value in (("metric", metric), ("feature_selection", feature_selection),
                       ("class_weight", class_weight), ("datetime_columns", datetime_columns)):
        if value:
            config[key] = value
    processor = DataProcessor.from_dict(config, df)
    processor.load_config()
    processor.read_data()
    paths = processor.paths_for(training=True)
    result = Just_Data_ML().Data_Train(
        Features_Cols=processor.Features_Cols, Data_df=processor.Data_df, Target_Col=processor.Target_Col,
        Models=processor.Models, Problem_Objective=processor.Problem_Objective,
        HyperParamter_Yes_or_No=processor.HyperParamter_Yes_or_No, Normalization_tech=processor.scaler,
        options=processor.options, paths=paths)
    remember_run(paths, processor.base_paths)
    return Model(processor.Problem_Objective, paths, result, result["features"])
