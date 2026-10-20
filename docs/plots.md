# Plots and reports

Install the extra and turn plots on:

```bash
pip install "JustDataML[plots]"
```

```yaml
plots: true
report: true      # one self-contained Output/report.html
explain: permutation   # or shap (needs JustDataML[shap])
```

| File in `Output/plots/` | Shows |
|---|---|
| `model_comparison.png` | Validation score of every model; the best is black |
| `feature_importance.png` | Importance in the best model (permutation importance when the model has none) |
| `selection_scores.png` | Feature selection scores; black bars are kept |
| `confusion_matrix.png`, `roc_curve.png`, `precision_recall_curve.png` | Classification only |
| `predicted_vs_actual.png`, `residuals.png` | Regression only |
| `target_distribution.png`, `correlation.png` | The data |
| `learning_curve.png` | Score as the training set grows |
| `permutation_importance.png` / `shap_importance.png` | With `explain` |

A plot that fails is skipped with a message; it never undoes a finished training run.
