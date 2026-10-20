import os
import re
from dataclasses import dataclass, field
from datetime import datetime

_RUN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunPaths:
    """
    Where a run keeps its model files (artifact_dir) and its reports (output_dir).
    With versioned runs both point at a per-run sub folder and base_output_dir keeps
    the folder where prediction files go.
    """

    artifact_dir: str = "artifact"
    output_dir: str = "Output"
    base_output_dir: str = ""

    @property
    def predictions_dir(self):
        return self.base_output_dir or self.output_dir

    @property
    def plots_dir(self):
        return os.path.join(self.output_dir, "plots")

    def artifact(self, name):
        return os.path.join(self.artifact_dir, name)

    def output(self, name):
        return os.path.join(self.output_dir, name)

    def run(self, run_name):
        if not _RUN_NAME.match(run_name):
            raise ValueError(f"Invalid run name '{run_name}'; use letters, digits, '.', '_' and '-'.")
        return RunPaths(os.path.join(self.artifact_dir, run_name),
                        os.path.join(self.output_dir, run_name), self.output_dir)

    def new_run(self):
        return self.run(datetime.now().strftime("%Y%m%d-%H%M%S"))

    def latest_run(self):
        """The newest run recorded by training, or the folder itself when none was recorded."""
        marker = os.path.join(self.artifact_dir, "latest.txt")
        if not os.path.exists(marker):
            raise FileNotFoundError(f"No versioned run found in '{self.artifact_dir}'. Train first.")
        with open(marker) as handle:
            return self.run(handle.read().strip())


@dataclass
class TrainOptions:
    """Settings of a training run beyond the data and models."""

    test_size: float = 0.2
    random_state: int = 14
    metric: str = ""                     # empty = accuracy (classification) or r2 (regression)
    cv_folds: int = 0                    # 0 = no cross-validation columns
    auto_features: bool = False          # features were not listed, take every non-target column
    feature_selection: list = field(default_factory=list)   # steps such as ["correlation", "importance"]
    top_k: int = 0                       # features to keep for ranking steps (0 = automatic)
    correlation_threshold: float = 0.95
    plots: bool = False
    explain: str = "none"                # none | permutation | shap
    report: bool = False
    datetime_columns: list = field(default_factory=list)
    max_onehot: int = 20                 # categorical columns with more values are target encoded
    class_weight: str = ""
