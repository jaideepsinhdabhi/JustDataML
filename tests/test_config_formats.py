import json

import pytest
import yaml

from conftest import write_config, write_yaml
from jdml.Configuration import DataProcessor
from jdml.Exception import CustomException


def load(path):
    dp = DataProcessor(path)
    dp.load_config()
    dp.read_data()
    return dp


def test_yaml_json_and_csv_give_the_same_config(workdir):
    from_csv = load(write_config(workdir / "c.csv"))
    from_yaml = load(write_yaml(workdir / "c.yaml"))
    (workdir / "c.json").write_text(json.dumps(yaml.safe_load((workdir / "c.yaml").read_text())))
    from_json = load(str(workdir / "c.json"))
    for other in (from_yaml, from_json):
        assert other.Features_Cols == from_csv.Features_Cols
        assert other.Target_Col == from_csv.Target_Col
        assert list(other.Models) == list(from_csv.Models)
        assert other.HyperParamter_Yes_or_No == "No"


def test_csv_config_warns_deprecated(workdir):
    cfg = write_config(workdir / "c.csv")
    with pytest.warns(DeprecationWarning, match="deprecated"):
        load(cfg)


def test_typed_values_and_aliases(workdir):
    cfg = write_yaml(workdir / "c.yml", tuning=True, test_size=0.3, seed=7, cv=3, plots=False, metric="F1")
    dp = load(cfg)
    assert dp.HyperParamter_Yes_or_No == "Yes"
    assert (dp.options.test_size, dp.options.random_state, dp.options.cv_folds, dp.options.plots) == (0.3, 7, 3, False)
    assert dp.options.metric == "f1"


def test_defaults_when_optional_keys_missing(workdir):
    path = workdir / "c.yaml"
    path.write_text("data: clf.csv\ntarget: label\nproblem: Classification\n")
    dp = load(str(path))
    assert dp.auto_features and dp.Features_Cols == ["f1", "f2", "color"]
    assert len(dp.Models) == 11 and dp.Normalization_tech == "StandardScaler"


def test_nested_feature_selection_block(workdir):
    cfg = write_yaml(workdir / "c.yaml", feature_selection={"method": ["correlation", "importance"], "top_k": 2})
    dp = load(cfg)
    assert dp.options.feature_selection == ["correlation", "importance"] and dp.options.top_k == 2


@pytest.mark.parametrize("override, message", [
    ({"nope": 1}, "Unknown setting"),
    ({"test_size": 5}, "out of range"),
    ({"cv_folds": 1}, "cv_folds"),
    ({"metric": "rmse"}, "not valid for Classification"),
    ({"feature_selection": "magic"}, "Unknown feature_selection"),
    ({"explain": "tea-leaves"}, "explain"),
    ({"class_weight": "heavy"}, "class_weight"),
    ({"artifact_dir": "../outside"}, "outside"),
    ({"output_dir": "."}, "sub folder"),
    ({"datetime_columns": ["missing"]}, "datetime_columns"),
])
def test_invalid_yaml_values(workdir, override, message):
    with pytest.raises(CustomException, match=message):
        load(write_yaml(workdir / "c.yaml", **override))


def test_yaml_is_loaded_safely(workdir):
    path = workdir / "evil.yaml"
    path.write_text("data: !!python/object/apply:os.system ['echo hacked']\n")
    with pytest.raises(CustomException):
        load(str(path))


def test_yaml_must_be_a_mapping(workdir):
    path = workdir / "list.yaml"
    path.write_text("- a\n- b\n")
    with pytest.raises(CustomException, match="mapping"):
        load(str(path))


def test_init_writes_template_and_refuses_overwrite(workdir, run_cli, capsys):
    run_cli("init", "-o", "mine.yaml")
    assert yaml.safe_load((workdir / "mine.yaml").read_text())["problem"] == "Classification"
    with pytest.raises(SystemExit):
        run_cli("init", "-o", "mine.yaml")
    run_cli("init", "-o", "mine.yaml", "--type", "regression", "--force")
    assert yaml.safe_load((workdir / "mine.yaml").read_text())["problem"] == "Regression"


def test_subcommands_train_and_predict(workdir, run_cli):
    cfg = write_yaml(workdir / "c.yaml")
    run_cli("train", "-C", cfg)
    run_cli("predict", "-C", cfg, "test_clf.csv", "-O", "viasub")
    assert (workdir / "Output" / "viasub.csv").exists()
