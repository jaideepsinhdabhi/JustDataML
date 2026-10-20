import pandas as pd

from conftest import write_config
from jdml.cli import Just_Data_ML
from jdml.Configuration import DataProcessor


def test_classification_train_and_predict(workdir, run_cli):
    cfg = write_config(workdir / "cfg.csv")
    run_cli("-C", cfg, "-T", "-P", "test_clf.csv", "-O", "pred")
    out = pd.read_csv(workdir / "Output" / "pred.csv")
    assert len(out) == 10
    assert set(out["Target_Out"]) <= {"a", "b", "c"}  # decoded back to the original labels
    for name in ("model.pkl", "preprocessor.pkl", "Targetlabel.pkl"):
        assert (workdir / "artifact" / name).exists()
        assert (workdir / "artifact" / f"{name}.sha256").exists()
    summary = pd.read_csv(workdir / "Output" / "Model_Summary.csv")
    assert "Accuracy_validation" in summary.columns and len(summary) == 2


def test_regression_train_and_predict(workdir, run_cli):
    cfg = write_config(
        workdir / "cfg.csv", Data_Name="reg.csv", Features_Cols=["x", "z"], Target_Col="y",
        Problem_Objective="Regression", Model_to_include=["Linear Regression"])
    run_cli("-C", cfg, "-T", "-P", "test_reg.csv")
    out = pd.read_csv(workdir / "Output" / "Target_outFile.csv")
    assert len(out) == 10 and out["Target_Out"].notna().all()


def test_predict_only_after_training_creates_output_dir(workdir, run_cli):
    cfg = write_config(workdir / "cfg.csv")
    run_cli("-C", cfg, "-T")
    (workdir / "Output").rename(workdir / "Output_old")
    run_cli("-C", cfg, "-P", "test_clf.csv")
    assert (workdir / "Output" / "Target_outFile.csv").exists()


def test_hyperparameter_search_runs(workdir, run_cli):
    cfg = write_config(workdir / "cfg.csv", HyperParamter_Yes_or_No="Yes", Model_to_include="GaussianNB")
    run_cli("-C", cfg, "-T")
    assert (workdir / "Output" / "Hyperparameter_stats.csv").exists()


def test_prediction_handles_unseen_category(workdir, run_cli):
    cfg = write_config(workdir / "cfg.csv")
    run_cli("-C", cfg, "-T")
    df = pd.read_csv(workdir / "test_clf.csv")
    df["color"] = "purple"
    df.to_csv(workdir / "unseen.csv", index=False)
    run_cli("-C", cfg, "-P", "unseen.csv", "-O", "unseen_out")
    assert len(pd.read_csv(workdir / "Output" / "unseen_out.csv")) == 10


def test_predict_does_not_mutate_input_and_checks_columns(workdir, run_cli):
    import pytest
    from jdml.Exception import CustomException

    cfg = write_config(workdir / "cfg.csv")
    run_cli("-C", cfg, "-T")
    dp = DataProcessor(cfg)
    dp.load_config()
    dp.read_data()
    frame = pd.read_csv(workdir / "test_clf.csv")
    before = frame.copy()
    result = Just_Data_ML().Predict_test("Classification", dp.Features_Cols, frame)
    assert "Target_Out" in result.columns
    pd.testing.assert_frame_equal(frame, before)
    with pytest.raises(CustomException, match="missing the feature columns"):
        Just_Data_ML().Predict_test("Classification", dp.Features_Cols, frame.drop(columns="f1"))


def test_training_does_not_mutate_shared_models(workdir, run_cli):
    from jdml.Utils import classfication_models

    cfg = write_config(workdir / "cfg.csv")
    run_cli("-C", cfg, "-T")
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    import pytest
    with pytest.raises(NotFittedError):
        check_is_fitted(classfication_models["Logistic Regression"])
