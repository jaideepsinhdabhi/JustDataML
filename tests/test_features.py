import json

import numpy as np
import pandas as pd
import pytest

from conftest import write_yaml
from jdml.components.Feature_Selection import select_features, unusable_columns


def wide_cfg(workdir, **overrides):
    base = dict(data="wide.csv", features="auto")
    base.update(overrides)
    return write_yaml(workdir / "c.yaml", **base)


def test_auto_features_drops_unusable_columns_and_keeps_signal(workdir, run_cli):
    run_cli("train", "-C", wide_cfg(workdir, feature_selection="importance", top_k=3))
    saved = json.loads((workdir / "artifact" / "features.json").read_text())["selected"]
    assert "f1" in saved or "f1_copy" in saved
    assert "row_id" not in saved and "const" not in saved
    report = pd.read_csv(workdir / "Output" / "selected_features.csv")
    reasons = dict(zip(report["feature"], report["reason"]))
    assert "constant" in reasons["const"] and "ID" in reasons["row_id"]
    assert len(saved) <= 3


def test_prediction_uses_saved_features_without_a_feature_list(workdir, run_cli):
    run_cli("train", "-C", wide_cfg(workdir, datetime_columns=["when"]))
    run_cli("predict", "-C", wide_cfg(workdir, datetime_columns=["when"]), "test_wide.csv", "-O", "wide")
    out = pd.read_csv(workdir / "Output" / "wide.csv")
    assert len(out) == 10 and set(out["Target_Out"]) <= {"a", "b", "c"}


def test_prediction_needs_only_the_selected_columns(workdir, run_cli):
    cfg = wide_cfg(workdir, feature_selection="importance", top_k=2)
    run_cli("train", "-C", cfg)
    saved = json.loads((workdir / "artifact" / "features.json").read_text())["selected"]
    pd.read_csv(workdir / "test_wide.csv")[saved].to_csv(workdir / "slim.csv", index=False)
    run_cli("predict", "-C", cfg, "slim.csv", "-O", "slim")
    assert len(pd.read_csv(workdir / "Output" / "slim.csv")) == 10


def test_explicit_features_are_kept_without_selection(workdir, run_cli):
    run_cli("train", "-C", write_yaml(workdir / "c.yaml"))
    saved = json.loads((workdir / "artifact" / "features.json").read_text())["selected"]
    assert saved == ["f1", "f2", "color"]


@pytest.mark.parametrize("method", ["variance", "correlation", "importance", "mutual_info", "rfe"])
def test_every_method_runs(workdir, run_cli, method):
    run_cli("train", "-C", wide_cfg(workdir, feature_selection=method, top_k=4))
    assert (workdir / "Output" / "selected_features.csv").exists()


def test_correlation_drops_duplicate_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=100)
    frame = pd.DataFrame({"a": x, "b": x * 3 + 1, "c": rng.normal(size=100)})
    kept, report = select_features(frame, (x > 0).astype(int), "Classification", ["correlation"])
    assert kept == ["a", "c"]
    assert report.set_index("feature").loc["b", "reason"] == "dropped by correlation"


def test_importance_uses_only_rows_it_is_given_and_keeps_signal():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    frame = pd.DataFrame({"signal": y + rng.normal(0, 0.2, 300), "n1": rng.normal(size=300), "n2": rng.normal(size=300)})
    kept, _ = select_features(frame, y, "Classification", ["importance"], top_k=1)
    assert kept == ["signal"]


def test_unusable_columns_detection():
    frame = pd.DataFrame({"const": [1] * 30, "ident": [f"x{i}" for i in range(30)], "ok": list(range(15)) * 2,
                          "floaty": np.linspace(0, 1, 30)})
    assert set(unusable_columns(frame)) == {"const", "ident"}


def test_selection_that_removes_everything_is_an_error():
    frame = pd.DataFrame({"const": [1] * 30})
    with pytest.raises(ValueError, match="removed every column"):
        select_features(frame, np.zeros(30, dtype=int), "Classification", [], auto=True)


def test_high_cardinality_and_dates_are_encoded(workdir, run_cli):
    cfg = write_yaml(workdir / "c.yaml", data="wide.csv", features=["f1", "city", "when"],
                     datetime_columns=["when"], max_onehot=10)
    run_cli("train", "-C", cfg)
    import joblib
    pre = joblib.load(workdir / "artifact" / "preprocessor.pkl")
    names = list(pre.get_feature_names_out())
    assert "when_year" in names and any(n.startswith("city") for n in names)
    assert not any(n.startswith("city_city") for n in names)  # target encoded, not one-hot
    run_cli("predict", "-C", cfg, "test_wide.csv", "-O", "enc")
    assert len(pd.read_csv(workdir / "Output" / "enc.csv")) == 10


def test_regression_auto_features(workdir, run_cli):
    cfg = write_yaml(workdir / "r.yaml", data="reg.csv", target="y", problem="Regression",
                     models=["Linear Regression"], features="auto", feature_selection="importance", top_k=1)
    run_cli("train", "-C", cfg)
    assert json.loads((workdir / "artifact" / "features.json").read_text())["selected"] == ["x"]
