import base64
import html
import json
import os
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import Plots
from .Logger import logging
from .Utils import GRID_SCORING, HIGHER_IS_BETTER


def write_metrics(ctx):
    """Write metrics.json: scores, features and the versions that produced the model."""
    import sklearn
    from . import __version__

    results, opts = ctx["results"], ctx["options"]
    payload = {
        "jdml_version": __version__,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "problem": ctx["problem"],
        "target": ctx["target"],
        "metric": results["metric"],
        "best_model": results["best_model_name"],
        "test_score": results["test_score"],
        "validation_scores": results["validation_scores"],
        "features": ctx["selected_features"],
        "rows": int(len(ctx["data_df"])),
        "random_state": opts.random_state,
        "test_size": opts.test_size,
        "cv_folds": opts.cv_folds,
        "model_summary_csv": ctx["paths"].output("Model_Summary.csv"),
    }
    path = ctx["paths"].output("metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _native_importance(model, n_features):
    values = getattr(model, "feature_importances_", None)
    if values is None and hasattr(model, "coef_"):
        coef = np.abs(np.asarray(model.coef_))
        values = coef.mean(axis=0) if coef.ndim > 1 else coef
    if values is not None and len(values) == n_features:
        return np.asarray(values, dtype=float)
    return None


def _permutation(results, random_state):
    from sklearn.inspection import permutation_importance
    found = permutation_importance(
        results["best_model"], results["X_test"], results["y_test"], n_repeats=5, random_state=random_state,
        scoring=GRID_SCORING.get(results["metric"]))
    return found.importances_mean


def make_plots(ctx):
    """Save the plots chosen by the options to the plots folder. Returns {title: path}."""
    results, opts, paths = ctx["results"], ctx["options"], ctx["paths"]
    classification = ctx["problem"] == "Classification"
    names = ctx["feature_names"]
    labels = list(ctx["label_encoder"].classes_) if classification else None
    made = {}

    def attempt(title, filename, func):
        try:
            path = func(os.path.join(paths.plots_dir, filename))
            if path:
                made[title] = path
        except Exception as e:  # a failed plot must not undo a finished training run
            logging.warning(f"Skipped plot '{title}': {e}")
            print(f" Skipped plot '{title}': {e}")

    attempt("Model comparison", "model_comparison.png", lambda p: Plots.model_comparison(
        results["validation_scores"], results["metric"], p, HIGHER_IS_BETTER[results["metric"]]))

    importance = _native_importance(results["best_model"], len(names))
    if importance is not None:
        attempt("Feature importance", "feature_importance.png",
                lambda p: Plots.importance_bars(names, importance, p, "Feature importance (best model)"))
    else:
        attempt("Feature importance", "feature_importance.png", lambda p: Plots.importance_bars(
            names, _permutation(results, opts.random_state), p, "Permutation importance (best model)",
            "drop in score when shuffled"))

    report = ctx.get("selection_report")
    if report is not None and report["score"].notna().any():
        scored = report.dropna(subset=["score"])
        kept_from = scored.loc[scored["kept"], "score_from"]
        if not kept_from.empty:  # scores of different steps are not comparable; plot the final ranking step
            scored = scored[scored["score_from"] == kept_from.mode().iloc[0]]
        attempt("Feature selection scores", "selection_scores.png", lambda p: Plots.importance_bars(
            list(scored["feature"]), scored["score"].to_numpy(), p, "Feature selection scores (black = kept)",
            "score", highlight=set(scored.loc[scored["kept"], "feature"])))

    if classification:
        attempt("Confusion matrix", "confusion_matrix.png", lambda p: Plots.confusion_matrix_plot(
            results["y_test"], results["y_pred"], labels, p))
        scores = Plots.class_scores(results["best_model"], results["X_test"])
        if scores is not None and scores.shape[1] == len(labels):
            attempt("ROC curve", "roc_curve.png",
                    lambda p: Plots.roc_curves(results["y_test"], scores, labels, p))
            attempt("Precision-recall curve", "precision_recall_curve.png",
                    lambda p: Plots.precision_recall_curves(results["y_test"], scores, labels, p))
    else:
        attempt("Predicted vs actual", "predicted_vs_actual.png",
                lambda p: Plots.predicted_vs_actual(results["y_test"], results["y_pred"], p))
        attempt("Residuals", "residuals.png", lambda p: Plots.residuals_plot(results["y_test"], results["y_pred"], p))

    attempt("Target distribution", "target_distribution.png", lambda p: Plots.target_distribution(
        ctx["data_df"][ctx["target"]], classification, p))
    numeric_cols = [c for c in ctx["selected_features"] if pd.api.types.is_numeric_dtype(ctx["data_df"][c])]
    attempt("Correlation", "correlation.png", lambda p: Plots.correlation_heatmap(
        ctx["data_df"][numeric_cols + [ctx["target"]]], p))

    from sklearn.base import clone
    attempt("Learning curve", "learning_curve.png", lambda p: Plots.learning_curve_plot(
        clone(results["best_model"]), results["X_train"], results["y_train"],
        GRID_SCORING.get(results["metric"]), p, opts.random_state))

    if opts.explain == "permutation":
        attempt("Permutation importance", "permutation_importance.png", lambda p: Plots.importance_bars(
            names, _permutation(results, opts.random_state), p, "Permutation importance (test set)",
            "drop in score when shuffled"))
    elif opts.explain == "shap":
        attempt("SHAP importance", "shap_importance.png", lambda p: Plots.shap_summary(
            results["best_model"], results["X_test"], names, p))
    return made


def write_html_report(ctx, plots):
    """One self-contained report.html with the metrics, selected features and plots."""
    results = ctx["results"]
    summary_path = ctx["paths"].output("Model_Summary.csv")
    summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else pd.DataFrame()
    selected = ", ".join(map(str, ctx["selected_features"]))
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>JDML report</title><style>body{font:15px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 16px;color:#1f2328}"
        "table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}td,th{border-bottom:1px solid #d8dee4;padding:4px 10px;text-align:right}"
        "td:first-child,th:first-child{text-align:left}img{max-width:100%;border:1px solid #d8dee4;margin:8px 0}h1,h2{font-weight:700}</style></head><body>",
        "<h1>JDML report</h1>",
        f"<p><b>Problem:</b> {html.escape(ctx['problem'])} &middot; <b>Target:</b> {html.escape(str(ctx['target']))} "
        f"&middot; <b>Rows:</b> {len(ctx['data_df'])}</p>",
        f"<p><b>Best model:</b> {html.escape(results['best_model_name'])} &middot; "
        f"<b>Test {html.escape(results['metric'])}:</b> {results['test_score']:.4f}</p>",
        f"<p><b>Features used ({len(ctx['selected_features'])}):</b> {html.escape(selected)}</p>",
        "<h2>Model summary</h2>", summary.round(4).to_html(index=False, border=0, escape=True),
    ]
    for title, path in plots.items():
        with open(path, "rb") as handle:
            encoded = base64.b64encode(handle.read()).decode("ascii")
        parts.append(f"<h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' src='data:image/png;base64,{encoded}'>")
    parts.append("</body></html>")
    path = ctx["paths"].output("report.html")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("".join(parts))
    return path
