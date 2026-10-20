import json

import pandas as pd
import pytest

import jdml
from conftest import write_yaml
from jdml.Exception import CustomException
from jdml.Utils import PROBLEM_METRICS, score_model


def clf_cfg(workdir, **overrides):
    return write_yaml(workdir / "c.yaml", **overrides)


def reg_cfg(workdir, **overrides):
    base = dict(data="reg.csv", target="y", problem="Regression", features=["x", "z"], models=["Linear Regression", "Decision Tree"])
    base.update(overrides)
    return write_yaml(workdir / "r.yaml", **base)


@pytest.mark.parametrize("metric", PROBLEM_METRICS["Classification"])
def test_classification_metrics(workdir, run_cli, metric):
    run_cli("train", "-C", clf_cfg(workdir, metric=metric))
    summary = pd.read_csv(workdir / "Output" / "Model_Summary.csv")
    assert set(summary["Metric"]) == {metric} and summary["Score_validation"].between(0, 1).all()


def test_regression_error_metrics_pick_the_lowest(workdir, run_cli):
    run_cli("train", "-C", reg_cfg(workdir, metric="rmse"))
    metrics = json.loads((workdir / "Output" / "metrics.json").read_text())
    assert metrics["metric"] == "rmse"
    assert metrics["best_model"] == min(metrics["validation_scores"], key=metrics["validation_scores"].get)


def test_cross_validation_columns(workdir, run_cli):
    run_cli("train", "-C", clf_cfg(workdir, cv_folds=3))
    summary = pd.read_csv(workdir / "Output" / "Model_Summary.csv")
    assert {"CV_accuracy_mean", "CV_accuracy_std"} <= set(summary.columns)


def test_class_weight_applies_where_supported(workdir):
    from jdml.Configuration import DataProcessor
    dp = DataProcessor(clf_cfg(workdir, class_weight="balanced"))
    dp.load_config()
    dp.read_data()
    assert dp.Models["Logistic Regression"].class_weight == "balanced"


def test_metrics_json_and_hist_gradient_boosting(workdir, run_cli):
    cfg = clf_cfg(workdir, models=["HistGradientBoosting Classifier"])
    run_cli("train", "-C", cfg)
    metrics = json.loads((workdir / "Output" / "metrics.json").read_text())
    assert metrics["best_model"] == "HistGradientBoosting Classifier"
    assert {"jdml_version", "scikit_learn", "features", "test_score", "random_state"} <= set(metrics)


def test_plots_and_report_classification(workdir, run_cli):
    run_cli("train", "-C", clf_cfg(workdir, plots=True, report=True, explain="permutation", cv_folds=0))
    plots = workdir / "Output" / "plots"
    for name in ("model_comparison", "feature_importance", "confusion_matrix", "roc_curve",
                 "precision_recall_curve", "target_distribution", "correlation", "learning_curve",
                 "permutation_importance"):
        assert (plots / f"{name}.png").stat().st_size > 1000, name
    report = (workdir / "Output" / "report.html").read_text()
    assert "data:image/png;base64" in report and "Model summary" in report


def test_plots_regression(workdir, run_cli):
    run_cli("train", "-C", reg_cfg(workdir, plots=True))
    plots = workdir / "Output" / "plots"
    for name in ("model_comparison", "feature_importance", "predicted_vs_actual", "residuals", "target_distribution"):
        assert (plots / f"{name}.png").stat().st_size > 1000, name


def test_no_plots_by_default(workdir, run_cli):
    run_cli("train", "-C", clf_cfg(workdir))
    assert not (workdir / "Output" / "plots").exists()


def test_versioned_runs(workdir, run_cli):
    cfg = clf_cfg(workdir, versioned_runs=True, run_name="exp1")
    run_cli("train", "-C", cfg)
    assert (workdir / "artifact" / "exp1" / "model.pkl").exists()
    assert (workdir / "Output" / "exp1" / "metrics.json").exists()
    run_cli("predict", "-C", cfg, "test_clf.csv", "-O", "p")
    assert (workdir / "Output" / "p.csv").exists()


def test_versioned_runs_pick_latest(workdir, run_cli):
    cfg = clf_cfg(workdir, versioned_runs=True)
    run_cli("train", "-C", cfg)
    run_dir = (workdir / "artifact" / "latest.txt").read_text().strip()
    assert (workdir / "artifact" / run_dir / "model.pkl").exists()
    run_cli("predict", "-C", cfg, "test_clf.csv")
    assert (workdir / "Output" / "Target_outFile.csv").exists()


def test_custom_artifact_and_output_dirs(workdir, run_cli):
    cfg = clf_cfg(workdir, artifact_dir="models/a", output_dir="reports")
    run_cli("train", "-C", cfg)
    run_cli("predict", "-C", cfg, "test_clf.csv", "-O", "q")
    assert (workdir / "models" / "a" / "model.pkl").exists() and (workdir / "reports" / "q.csv").exists()
    assert not (workdir / "artifact").exists()


def test_python_api(workdir):
    df = pd.read_csv(workdir / "Data" / "wide.csv")
    model = jdml.fit(df.drop(columns=["row_id", "const"]), target="label", models=["Logistic Regression"],
                     feature_selection="importance", top_k=2)
    assert model.problem == "Classification" and len(model.features) <= 2
    preds = model.predict(pd.read_csv(workdir / "test_wide.csv"))
    assert len(preds) == 10 and set(preds) <= {"a", "b", "c"}
    assert len(model.report()) == 1
    assert "Logistic Regression" in repr(model)
    again = jdml.Model.load("Classification")
    assert list(again.predict(pd.read_csv(workdir / "test_wide.csv"))) == list(preds)


def test_python_api_infers_regression(workdir):
    df = pd.read_csv(workdir / "Data" / "reg.csv")
    model = jdml.fit(df, target="y", models=["Linear Regression"])
    assert model.problem == "Regression" and model.score > 0.9


def test_python_api_does_not_modify_the_frame(workdir):
    df = pd.read_csv(workdir / "Data" / "reg.csv")
    before = df.copy()
    jdml.fit(df, target="y", models=["Linear Regression"])
    pd.testing.assert_frame_equal(df, before)


def test_unknown_metric_for_score_model():
    with pytest.raises(ValueError):
        score_model(None, None, None, "bogus")


def test_missing_plot_dependency_message(workdir, monkeypatch):
    import jdml.Configuration as configuration
    monkeypatch.setattr(configuration, "plotting_available", lambda: False)
    dp = configuration.DataProcessor(clf_cfg(workdir, plots=True))
    dp.load_config()
    with pytest.raises(CustomException, match="JustDataML\\[plots\\]"):
        dp.read_data()


@pytest.mark.parametrize("higher, best", [(True, "b"), (False, "a")])
def test_model_comparison_highlights_the_best_model(tmp_path, higher, best):
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    from jdml import Plots

    captured = {}
    original = Plots._save
    Plots._save = lambda fig, path: captured.setdefault("ax", fig.axes[0]) and path
    try:
        Plots.model_comparison({"a": 0.2, "b": 0.9}, "m", str(tmp_path / "x.png"), higher_is_better=higher)
    finally:
        Plots._save = original
    ink = to_rgb(Plots.INK)
    black_widths = [bar.get_width() for bar in captured["ax"].patches if bar.get_facecolor()[:3] == ink]
    assert black_widths == [{"b": 0.9, "a": 0.2}[best]]
    plt.close("all")


@pytest.mark.parametrize("package, model", [("lightgbm", "LightGBM Classifier"), ("catboost", "CatBoost Classifier")])
def test_optional_boosting_models(workdir, run_cli, package, model):
    pytest.importorskip(package)
    cfg = clf_cfg(workdir, models=[model])
    run_cli("train", "-C", cfg)
    run_cli("predict", "-C", cfg, "test_clf.csv", "-O", package)
    assert len(pd.read_csv(workdir / "Output" / f"{package}.csv")) == 10


def test_optional_model_error_names_the_extra(workdir, monkeypatch):
    import jdml.Configuration as configuration
    monkeypatch.delitem(configuration.classfication_models, "LightGBM Classifier", raising=False)
    dp = configuration.DataProcessor(clf_cfg(workdir, models=["LightGBM Classifier"]))
    dp.load_config()
    with pytest.raises(CustomException, match="JustDataML\\[lightgbm\\]"):
        dp.read_data()


def test_shap_importance_plot(workdir, run_cli):
    pytest.importorskip("shap")
    run_cli("train", "-C", clf_cfg(workdir, models=["Random Forest Classifier"], explain="shap", plots=True))
    assert (workdir / "Output" / "plots" / "shap_importance.png").exists()


def test_shap_missing_is_a_clear_error(workdir, monkeypatch):
    import importlib.util
    import jdml.Configuration as configuration
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    dp = configuration.DataProcessor(clf_cfg(workdir, explain="shap"))
    dp.load_config()
    with pytest.raises(CustomException, match="JustDataML\\[shap\\]"):
        dp.read_data()
