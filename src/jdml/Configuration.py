import sys

import pandas as pd
from sklearn.base import clone
from .Exception import CustomException
from .Utils import LEGACY_MODEL_NAMES, Normalization_method, ReadConfig, classfication_models, regressor_models, safe_join


class DataProcessor:
    def __init__(self, config_file):
        self.config_file = config_file

    def load_config(self):
        self.DataConfig = ReadConfig(self.config_file)

    def _first(self, key):
        values = self.DataConfig.get(key)
        if not values:
            raise ValueError(f"'{key}' is missing or empty in the config file '{self.config_file}'.")
        return values[0]

    def read_data(self):
        try:
            for key in ("Data_Name", "Features_Cols", "Target_Col", "Normalization_tech",
                        "Model_to_include", "Problem_Objective", "HyperParamter_Yes_or_No"):
                self._first(key)

            self.Data_df = pd.read_csv(safe_join("Data", self._first("Data_Name")))
            self.Features_Cols = self.DataConfig["Features_Cols"]
            self.Target_Col = self.DataConfig["Target_Col"]
            self.Normalization_tech = self._first("Normalization_tech")
            self.Model_to_include = [LEGACY_MODEL_NAMES.get(name, name) for name in self.DataConfig["Model_to_include"]]
            self.Problem_Objective = self._first("Problem_Objective")
            self.HyperParamter_Yes_or_No = self._first("HyperParamter_Yes_or_No")

            missing = [col for col in self.Features_Cols + self.Target_Col if col not in self.Data_df.columns]
            if missing:
                raise ValueError(f"Columns {missing} from the config are not in the data file.")
            if len(self.Target_Col) != 1:
                raise ValueError("Target_Col must contain exactly one column.")
            if self.Target_Col[0] in self.Features_Cols:
                raise ValueError(f"Target column '{self.Target_Col[0]}' must not be listed in Features_Cols.")
            if self.HyperParamter_Yes_or_No not in ("Yes", "No"):
                raise ValueError("HyperParamter_Yes_or_No must be 'Yes' or 'No'.")

            if self.Problem_Objective == "Regression":
                available = regressor_models
            elif self.Problem_Objective == "Classification":
                available = classfication_models
            else:
                raise ValueError(f"Problem_Objective '{self.Problem_Objective}' is not valid; use 'Regression' or 'Classification'.")

            if self.Model_to_include[0] == "ALL":
                selected = list(available)
            else:
                unknown = [name for name in self.Model_to_include if name not in available]
                if unknown:
                    raise ValueError(f"Unknown models {unknown} for {self.Problem_Objective}. Available: {list(available)}")
                selected = self.Model_to_include
            # Clones, so training never mutates the shared module-level estimators.
            self.Models = {name: clone(available[name]) for name in selected}

            if self.Normalization_tech not in Normalization_method:
                raise ValueError(f"{self.Normalization_tech} is not a valid normalization method. Use one of {list(Normalization_method)}.")
            self.scaler = clone(Normalization_method[self.Normalization_tech])

        except Exception as e:
            raise CustomException(e,sys)

