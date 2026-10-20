CLASSIFICATION = """\
# JDML config. Run: JDML train -C this_file.yaml   then   JDML predict -C this_file.yaml new_data.csv
data: bezdekIris.data            # file inside the Data folder
target: class
problem: Classification          # Classification or Regression
features: auto                   # auto = choose useful columns by itself, or list them: [col_a, col_b]
feature_selection: importance    # none, variance, correlation, importance, mutual_info, rfe (or a list)
# top_k: 10                      # keep at most this many features
models: ALL                      # ALL or a list such as [Random Forest Classifier, XGBClassifier]
scaler: StandardScaler           # StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
tuning: false                    # true = grid search hyperparameters (slower)
metric: accuracy                 # accuracy, f1, precision, recall, roc_auc
cv_folds: 0                      # 0 = off, or 3-10 for cross-validation columns in the summary
class_weight: balanced           # helps with imbalanced classes
plots: true                      # saves charts to Output/plots (pip install "JustDataML[plots]")
report: false                    # true = also write Output/report.html
"""

REGRESSION = """\
# JDML config. Run: JDML train -C this_file.yaml   then   JDML predict -C this_file.yaml new_data.csv
data: CASP.csv                   # file inside the Data folder
target: RMSD
problem: Regression              # Classification or Regression
features: auto                   # auto = choose useful columns by itself, or list them: [col_a, col_b]
feature_selection: [correlation, importance]   # none, variance, correlation, importance, mutual_info, rfe
# top_k: 10                      # keep at most this many features
models: [Linear Regression, Random Forest, HistGradientBoosting Regressor]   # or ALL
scaler: StandardScaler
tuning: false
metric: r2                       # r2, rmse, mae
cv_folds: 0
plots: true
report: false
"""
