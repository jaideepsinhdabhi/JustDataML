import importlib.util
import os
import sys

import pandas as pd
from sklearn.base import clone
from .Exception import CustomException
from .Options import RunPaths, TrainOptions
from .Plots import plotting_available
from .components.Feature_Selection import METHODS
from .Utils import (CORE_CLASSIFIERS, CORE_REGRESSORS, DEFAULT_METRIC, LEGACY_MODEL_NAMES, OPTIONAL_MODEL_EXTRAS,
                    PROBLEM_METRICS, Normalization_method, ReadConfig, classfication_models, config_from_mapping,
                    regressor_models, safe_join)

AUTO = "auto"
TRUE_WORDS = ("true", "yes", "1", "on")
FALSE_WORDS = ("false", "no", "0", "off")


class DataProcessor:
    """
    Reads and validates a config file (or a dict, for the Python API) and loads the data.
    After read_data() the settings are on the object: Data_df, Features_Cols, Target_Col, Models,
    Problem_Objective, scaler, options (a TrainOptions) and base_paths (a RunPaths).
    """

    def __init__(self, config_file=None, config=None, data_df=None):
        self.config_file = config_file
        self._config = config
        self._data_df = data_df

    @classmethod
    def from_dict(cls, config, data_df):
        """Build from a dict of settings (same names as the YAML file) and a DataFrame."""
        return cls(config=config, data_df=data_df)

    def load_config(self):
        if self._config is not None:
            self.DataConfig = config_from_mapping(self._config, "<dict>")
        else:
            self.DataConfig = ReadConfig(self.config_file)

    def _first(self, key):
        values = self.DataConfig.get(key)
        if not values:
            raise ValueError(f"'{key}' is missing or empty in the config file '{self.config_file}'.")
        return values[0]

    def _optional(self, key, default=""):
        values = self.DataConfig.get(key)
        return values[0] if values else default

    def _flag(self, key, default=False):
        value = self._optional(key, "").lower()
        if not value:
            return default
        if value in TRUE_WORDS:
            return True
        if value in FALSE_WORDS:
            return False
        raise ValueError(f"'{key}' must be true or false, not '{value}'.")

    def _number(self, key, kind, default, low=None, high=None, exclusive=False):
        raw = self._optional(key, "")
        if raw == "":
            return default
        try:
            value = kind(raw)
        except ValueError:
            raise ValueError(f"'{key}' must be a number, not '{raw}'.") from None
        too_low = low is not None and (value <= low if exclusive else value < low)
        too_high = high is not None and (value >= high if exclusive else value > high)
        if too_low or too_high:
            raise ValueError(f"'{key}' = {value} is out of range.")
        return value

    def _directory(self, key, default):
        value = self._optional(key, default)
        relative = os.path.normpath(safe_join(".", value))  # must stay inside the working directory
        if relative == ".":
            raise ValueError(f"'{key}' must be a sub folder, not the working directory itself.")
        return relative

    def _read_options(self, problem):
        """Validate the optional settings and return them as a TrainOptions."""
        opts = TrainOptions()
        opts.test_size = self._number("test_size", float, opts.test_size, 0, 1, exclusive=True)
        opts.random_state = self._number("random_state", int, opts.random_state)
        opts.cv_folds = self._number("cv_folds", int, opts.cv_folds, 0)
        if opts.cv_folds == 1:
            raise ValueError("'cv_folds' must be 0 (off) or at least 2.")
        opts.top_k = self._number("top_k", int, opts.top_k, 0)
        opts.correlation_threshold = self._number("correlation_threshold", float, opts.correlation_threshold, 0, 1)
        opts.max_onehot = self._number("max_onehot", int, opts.max_onehot, 1)

        opts.metric = self._optional("metric", "").lower()
        if opts.metric and opts.metric not in PROBLEM_METRICS[problem]:
            raise ValueError(f"metric '{opts.metric}' is not valid for {problem}; use one of {PROBLEM_METRICS[problem]}.")
        opts.feature_selection = [m.lower() for m in self.DataConfig.get("feature_selection", [])]
        unknown = [m for m in opts.feature_selection if m not in METHODS]
        if unknown:
            raise ValueError(f"Unknown feature_selection {unknown}; use one of {list(METHODS)}.")
        opts.explain = self._optional("explain", "none").lower()
        if opts.explain not in ("none", "permutation", "shap"):
            raise ValueError("'explain' must be none, permutation or shap.")
        opts.report = self._flag("report")
        opts.plots = self._flag("plots") or opts.report
        if (opts.plots or opts.explain != "none") and not plotting_available():
            raise ValueError('Plots need matplotlib: pip install "JustDataML[plots]"')
        if opts.explain == "shap" and importlib.util.find_spec("shap") is None:
            raise ValueError('explain: shap needs the shap package: pip install "JustDataML[shap]"')
        opts.datetime_columns = list(self.DataConfig.get("datetime_columns", []))
        opts.class_weight = self._optional("class_weight", "").lower()
        if opts.class_weight not in ("", "none", "balanced"):
            raise ValueError("'class_weight' must be balanced or none.")
        return opts

    def read_data(self):
        try:
            required = ["Target_Col", "Problem_Objective"]
            if self._data_df is None:
                required.insert(0, "Data_Name")
            for key in required:
                self._first(key)

            if self._data_df is not None:
                self.Data_df = self._data_df.copy()
            else:
                self.Data_df = pd.read_csv(safe_join("Data", self._first("Data_Name")))
            self.Target_Col = self.DataConfig["Target_Col"]
            self.Normalization_tech = self._optional("Normalization_tech", "StandardScaler")
            self.Model_to_include = [LEGACY_MODEL_NAMES.get(name, name) for name in self.DataConfig.get("Model_to_include") or ["ALL"]]
            self.Problem_Objective = self._first("Problem_Objective")
            self.HyperParamter_Yes_or_No = self._optional("HyperParamter_Yes_or_No", "No")

            if len(self.Target_Col) != 1:
                raise ValueError("Target_Col must contain exactly one column.")
            if self.Target_Col[0] not in self.Data_df.columns:
                raise ValueError(f"Columns {self.Target_Col} from the config are not in the data file.")

            listed = self.DataConfig.get("Features_Cols") or [AUTO]
            self.auto_features = [f.lower() for f in listed] == [AUTO]
            if self.auto_features:
                self.Features_Cols = [c for c in self.Data_df.columns if c != self.Target_Col[0]]
            else:
                self.Features_Cols = listed
                missing = [col for col in self.Features_Cols if col not in self.Data_df.columns]
                if missing:
                    raise ValueError(f"Columns {missing} from the config are not in the data file.")
                if self.Target_Col[0] in self.Features_Cols:
                    raise ValueError(f"Target column '{self.Target_Col[0]}' must not be listed in Features_Cols.")
            if self.HyperParamter_Yes_or_No not in ("Yes", "No"):
                raise ValueError("HyperParamter_Yes_or_No must be 'Yes' or 'No'.")

            if self.Problem_Objective == "Regression":
                available, core = regressor_models, CORE_REGRESSORS
            elif self.Problem_Objective == "Classification":
                available, core = classfication_models, CORE_CLASSIFIERS
            else:
                raise ValueError(f"Problem_Objective '{self.Problem_Objective}' is not valid; use 'Regression' or 'Classification'.")

            self.options = self._read_options(self.Problem_Objective)
            self.options.auto_features = self.auto_features
            missing_dates = [c for c in self.options.datetime_columns if c not in self.Data_df.columns]
            if missing_dates:
                raise ValueError(f"datetime_columns {missing_dates} are not in the data file.")
            if self.options.datetime_columns:
                for col in self.options.datetime_columns:
                    self.Data_df[col] = pd.to_datetime(self.Data_df[col], errors="coerce", format="mixed")

            if self.Model_to_include[0] == "ALL":
                selected = list(core)
            else:
                for name in self.Model_to_include:
                    if name in OPTIONAL_MODEL_EXTRAS and name not in available:
                        extra = OPTIONAL_MODEL_EXTRAS[name]
                        raise ValueError(f"Model '{name}' needs the {extra} package: pip install \"JustDataML[{extra}]\"")
                unknown = [name for name in self.Model_to_include if name not in available]
                if unknown:
                    raise ValueError(f"Unknown models {unknown} for {self.Problem_Objective}. Available: {list(available)}")
                selected = self.Model_to_include
            # Clones, so training never mutates the shared module-level estimators.
            self.Models = {name: clone(available[name]) for name in selected}
            if self.options.class_weight == "balanced" and self.Problem_Objective == "Classification":
                for model in self.Models.values():
                    if "class_weight" in model.get_params():
                        model.set_params(class_weight="balanced")

            if self.Normalization_tech not in Normalization_method:
                raise ValueError(f"{self.Normalization_tech} is not a valid normalization method. Use one of {list(Normalization_method)}.")
            self.scaler = clone(Normalization_method[self.Normalization_tech])

            self.base_paths = RunPaths(self._directory("artifact_dir", "artifact"), self._directory("output_dir", "Output"))
            self.versioned = self._flag("versioned_runs")
            self.run_name = self._optional("run_name", "")
            self.metric = self.options.metric or DEFAULT_METRIC[self.Problem_Objective]

        except Exception as e:
            raise CustomException(e,sys)

    def paths_for(self, training):
        """
        The folders to use. With versioned_runs, training starts a new run folder (named by
        run_name, or the current time) and prediction uses run_name or the latest run.
        """
        if not self.versioned:
            return self.base_paths
        if self.run_name:
            return self.base_paths.run(self.run_name)
        return self.base_paths.new_run() if training else self.base_paths.latest_run()


def remember_run(paths, base_paths):
    """Record the run folder just trained as the latest one (versioned runs only)."""
    if paths.artifact_dir != base_paths.artifact_dir:
        os.makedirs(base_paths.artifact_dir, exist_ok=True)
        with open(os.path.join(base_paths.artifact_dir, "latest.txt"), "w") as handle:
            handle.write(os.path.basename(paths.artifact_dir))
