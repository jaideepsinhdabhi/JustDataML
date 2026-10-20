import pytest

from conftest import write_config
from jdml.Configuration import DataProcessor
from jdml.Exception import CustomException


def load(path):
    dp = DataProcessor(path)
    dp.load_config()
    dp.read_data()
    return dp


@pytest.mark.parametrize("override, message", [
    ({"Problem_Objective": "Clustering"}, "not valid"),
    ({"Normalization_tech": "NopeScaler"}, "not a valid normalization"),
    ({"Model_to_include": "Imaginary Model"}, "Unknown models"),
    ({"Features_Cols": ["f1", "label"]}, "must not be listed in Features_Cols"),
    ({"Features_Cols": ["f1", "missing_col"]}, "not in the data file"),
    ({"HyperParamter_Yes_or_No": "Maybe"}, "'Yes' or 'No'"),
])
def test_invalid_config_raises_clear_error(workdir, override, message):
    cfg = write_config(workdir / "cfg.csv", **override)
    with pytest.raises(CustomException, match=message):
        load(cfg)


def test_missing_config_file(workdir):
    with pytest.raises(CustomException, match="not found"):
        load(str(workdir / "nope.csv"))


def test_missing_key(workdir):
    path = workdir / "cfg.csv"
    path.write_text("Data_Name,clf.csv\n")
    with pytest.raises(CustomException, match="missing or empty"):
        load(str(path))


def test_blank_rows_and_whitespace_are_tolerated(workdir):
    cfg = write_config(workdir / "cfg.csv")
    text = (workdir / "cfg.csv").read_text().replace("Features_Cols,", "\n,,\nFeatures_Cols, ")
    (workdir / "cfg.csv").write_text(text)
    assert load(cfg).Features_Cols == ["f1", "f2", "color"]


def test_legacy_ridge_name_and_all_models(workdir):
    cfg = write_config(workdir / "cfg.csv", Model_to_include="Ridge Classifcaiton")
    assert list(load(cfg).Models) == ["Ridge Classification"]
    cfg = write_config(workdir / "cfg.csv", Model_to_include="ALL")
    assert len(load(cfg).Models) == 10


def test_models_are_clones_of_module_level_ones(workdir):
    from jdml.Utils import classfication_models

    dp = load(write_config(workdir / "cfg.csv"))
    assert dp.Models["Logistic Regression"] is not classfication_models["Logistic Regression"]
