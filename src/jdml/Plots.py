"""
Plot functions. Each one takes plain data, saves a PNG and returns its path.
Needs matplotlib: pip install "JustDataML[plots]".
"""
import os

import numpy as np
import pandas as pd

INK = "#1f2328"
MUTED = "#6e7781"
GRID = "#d8dee4"
SERIES = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]  # Okabe-Ito, fixed order
MAX_CLASSES_DRAWN = 8
MAX_FEATURES_DRAWN = 20


def plotting_available():
    try:
        import matplotlib  # noqa: F401
        return True
    except ImportError:
        return False


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Plots need matplotlib: pip install "JustDataML[plots]"') from e
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 150, "font.size": 10,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    })
    return plt


def _save(fig, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    _plt().close(fig)
    return path


def model_comparison(scores, metric, path, higher_is_better=True):
    """Horizontal bars of every model's validation score; the winner is drawn in black."""
    plt = _plt()
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=higher_is_better)  # best first
    names = [k for k, _ in ordered][::-1]
    values = [v for _, v in ordered][::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.45 * len(names) + 1)))
    colors = [MUTED] * (len(names) - 1) + [INK]
    ax.barh(names, values, color=colors, height=0.6)
    for y, v in enumerate(values):
        ax.text(v, y, f" {v:.3f}", va="center", color=INK, fontsize=9)
    ax.set_xlabel(f"{metric} (validation)" + ("" if higher_is_better else ", lower is better"))
    ax.set_title("Model comparison")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def importance_bars(names, values, path, title="Feature importance", xlabel="importance", highlight=None):
    """Top features by value, largest at the top."""
    plt = _plt()
    order = np.argsort(values)[::-1][:MAX_FEATURES_DRAWN][::-1]
    shown = [str(names[i]) for i in order]
    vals = [float(values[i]) for i in order]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(shown) + 1)))
    colors = [INK if highlight is None or n in highlight else GRID for n in shown]
    ax.barh(shown, vals, color=colors, height=0.6)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def confusion_matrix_plot(y_true, y_pred, labels, path):
    from sklearn.metrics import confusion_matrix
    plt = _plt()
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    size = max(4.5, 0.7 * len(labels) + 2)
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(matrix, cmap="Greys")
    ax.grid(False)
    ax.set_xticks(range(len(labels)), [str(label) for label in labels], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), [str(label) for label in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix")
    top = matrix.max() or 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > top / 2 else INK)
    return _save(fig, path)


def class_scores(model, X):
    """Per-class scores (probabilities if the model has them) as an (n, classes) array, or None."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return np.c_[-scores, scores] if scores.ndim == 1 else scores
    return None


def _curve_targets(y_true, scores, labels):
    """Yield (name, binary truth, score) for the positive class or each class one-vs-rest."""
    if len(labels) == 2:
        yield str(labels[1]), (np.asarray(y_true) == 1).astype(int), scores[:, 1]
    else:
        for index in range(min(len(labels), MAX_CLASSES_DRAWN)):
            yield str(labels[index]), (np.asarray(y_true) == index).astype(int), scores[:, index]


def roc_curves(y_true, scores, labels, path):
    from sklearn.metrics import auc, roc_curve
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1)
    for color, (name, truth, score) in zip(SERIES, _curve_targets(y_true, scores, labels)):
        if truth.min() == truth.max():
            continue
        fpr, tpr, _ = roc_curve(truth, score)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC {auc(fpr, tpr):.2f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="lower right")
    return _save(fig, path)


def precision_recall_curves(y_true, scores, labels, path):
    from sklearn.metrics import average_precision_score, precision_recall_curve
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for color, (name, truth, score) in zip(SERIES, _curve_targets(y_true, scores, labels)):
        if truth.min() == truth.max():
            continue
        precision, recall, _ = precision_recall_curve(truth, score)
        ax.plot(recall, precision, color=color, linewidth=2,
                label=f"{name} (AP {average_precision_score(truth, score):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve")
    ax.legend(loc="lower left")
    return _save(fig, path)


def predicted_vs_actual(y_true, y_pred, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(y_true, y_pred, s=18, color=SERIES[0], alpha=0.7, edgecolors="white", linewidths=0.5)
    low, high = min(np.min(y_true), np.min(y_pred)), max(np.max(y_true), np.max(y_pred))
    ax.plot([low, high], [low, high], color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs actual")
    return _save(fig, path)


def residuals_plot(y_true, y_pred, path):
    plt = _plt()
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={"width_ratios": [3, 2]})
    ax1.scatter(y_pred, residuals, s=18, color=SERIES[0], alpha=0.7, edgecolors="white", linewidths=0.5)
    ax1.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residual (actual - predicted)")
    ax1.set_title("Residuals")
    ax2.hist(residuals, bins=20, color=MUTED, edgecolor="white")
    ax2.set_xlabel("Residual")
    ax2.set_title("Distribution")
    return _save(fig, path)


def target_distribution(y, classification, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    series = pd.Series(y)
    if classification:
        counts = series.value_counts().sort_index()
        ax.bar([str(k) for k in counts.index], counts.values, color=MUTED, width=0.6)
        ax.set_ylabel("Rows")
        ax.grid(axis="x", visible=False)
    else:
        ax.hist(series.dropna(), bins=30, color=MUTED, edgecolor="white")
        ax.set_ylabel("Rows")
    ax.set_title(f"Target distribution: {series.name}" if series.name is not None else "Target distribution")
    return _save(fig, path)


def correlation_heatmap(frame, path):
    """Correlation of the numeric columns, or None when there are fewer than two."""
    numeric = frame.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return None
    numeric = numeric.iloc[:, :30]
    plt = _plt()
    corr = numeric.corr()
    size = max(5, 0.5 * len(corr) + 2)
    fig, ax = plt.subplots(figsize=(size, size * 0.85))
    image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    if len(corr) <= 12:
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(corr.iat[i, j]) > 0.6 else INK)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Correlation")
    return _save(fig, path)


def learning_curve_plot(model, X, y, scoring, path, random_state=14):
    from sklearn.model_selection import learning_curve
    plt = _plt()
    sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=3, scoring=scoring, train_sizes=np.linspace(0.2, 1.0, 5), random_state=random_state, shuffle=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, scores, color in (("Training", train_scores, INK), ("Cross-validation", val_scores, SERIES[0])):
        mean, std = scores.mean(axis=1), scores.std(axis=1)
        ax.plot(sizes, mean, color=color, linewidth=2, marker="o", markersize=5, label=name)
        ax.fill_between(sizes, mean - std, mean + std, color=color, alpha=0.12)
    ax.set_xlabel("Training rows")
    ax.set_ylabel(scoring or "score")
    ax.set_title("Learning curve")
    ax.legend(loc="lower right")
    return _save(fig, path)


def shap_summary(model, X, names, path, max_rows=200):
    """Mean absolute SHAP value per feature (needs the shap extra)."""
    try:
        import shap
    except ImportError as e:
        raise ImportError('SHAP plots need shap: pip install "JustDataML[shap]"') from e
    sample = X[:max_rows]
    explainer = shap.Explainer(model, sample, feature_names=list(names))
    values = np.abs(explainer(sample).values)
    while values.ndim > 2:
        values = values.mean(axis=-1)
    return importance_bars(names, values.mean(axis=0), path, title="SHAP importance", xlabel="mean |SHAP value|")
