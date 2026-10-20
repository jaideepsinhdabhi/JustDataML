import sys

import numpy as np
import pandas as pd
import pytest

from jdml.cli import main


def write_config(path, **overrides):
    cfg = {
        "Data_Name": ["clf.csv"],
        "Features_Cols": ["f1", "f2", "color"],
        "Target_Col": ["label"],
        "Problem_Objective": ["Classification"],
        "Normalization_tech": ["StandardScaler"],
        "Model_to_include": ["Logistic Regression", "Decision Tree Classifier"],
        "HyperParamter_Yes_or_No": ["No"],
    }
    cfg.update({k: v if isinstance(v, list) else [v] for k, v in overrides.items()})
    path.write_text("\n".join(",".join([k, *v]) for k, v in cfg.items()) + "\n")
    return str(path)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Empty project dir (cwd) with a Data/ folder holding a classification and a regression CSV."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Data").mkdir()
    rng = np.random.default_rng(0)
    n = 150
    label = rng.integers(0, 3, n)
    clf = pd.DataFrame({
        "f1": label * 2.0 + rng.normal(0, 0.3, n),
        "f2": rng.normal(size=n),
        "color": rng.choice(["red", "green", "blue"], n),
        "label": np.array(["a", "b", "c"])[label],
    })
    clf.to_csv(tmp_path / "Data" / "clf.csv", index=False)
    x = rng.normal(size=n)
    reg = pd.DataFrame({"x": x, "z": rng.normal(size=n), "y": 3 * x + rng.normal(0, 0.1, n)})
    reg.to_csv(tmp_path / "Data" / "reg.csv", index=False)
    clf.drop(columns="label").head(10).to_csv(tmp_path / "test_clf.csv", index=False)
    reg.drop(columns="y").head(10).to_csv(tmp_path / "test_reg.csv", index=False)
    return tmp_path


@pytest.fixture
def run_cli(monkeypatch):
    def _run(*args):
        monkeypatch.setattr(sys, "argv", ["jdml", *args])
        main()
    return _run
