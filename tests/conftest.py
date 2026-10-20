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


def write_yaml(path, **overrides):
    import yaml

    cfg = {
        "data": "clf.csv", "features": ["f1", "f2", "color"], "target": "label",
        "problem": "Classification", "models": ["Logistic Regression", "Decision Tree Classifier"],
    }
    cfg.update(overrides)
    path.write_text(yaml.safe_dump(cfg))
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
    # Extra columns for feature selection tests: an ID, a constant, noise, and a copy of f1.
    wide = clf.copy()
    wide.insert(0, "row_id", [f"id{i}" for i in range(n)])
    wide["const"] = 1
    wide["noise"] = rng.normal(size=n)
    wide["f1_copy"] = wide["f1"] * 2 + 1
    wide["when"] = pd.date_range("2024-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    wide["city"] = [f"city{i % 40}" for i in range(n)]
    wide.to_csv(tmp_path / "Data" / "wide.csv", index=False)
    wide.drop(columns="label").head(10).to_csv(tmp_path / "test_wide.csv", index=False)
    clf.drop(columns="label").head(10).to_csv(tmp_path / "test_clf.csv", index=False)
    reg.drop(columns="y").head(10).to_csv(tmp_path / "test_reg.csv", index=False)
    return tmp_path


@pytest.fixture
def run_cli(monkeypatch):
    def _run(*args):
        monkeypatch.setattr(sys, "argv", ["jdml", *args])
        main()
    return _run
