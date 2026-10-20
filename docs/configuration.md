# Configuration reference

A config is a YAML (`.yaml`, `.yml`) or JSON file. Unknown settings are an error, so typos are caught. The old CSV format is deprecated but still read.

| Setting (aliases) | Default | Values |
|---|---|---|
| `data` (`Data_Name`) | required | File name inside the `Data/` folder |
| `target` (`Target_Col`) | required | One column name |
| `problem` (`Problem_Objective`) | required | `Classification`, `Regression` |
| `features` (`Features_Cols`) | `auto` | `auto` or a list |
| `feature_selection` | none; `importance` with `auto` | `none`, `variance`, `correlation`, `importance`, `mutual_info`, `rfe`, or a list of them. Can also be a block with `method`, `top_k`, `correlation_threshold` |
| `top_k` | 0 | Features to keep for ranking methods (0 = automatic) |
| `correlation_threshold` | 0.95 | 0 to 1 |
| `models` (`Model_to_include`) | `ALL` | `ALL` = core models, or names (see README) |
| `scaler` (`Normalization_tech`) | `StandardScaler` | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` |
| `tuning` (`HyperParamter_Yes_or_No`) | `false` | `true`/`false` (`Yes`/`No` in CSV) |
| `metric` | `accuracy`, `r2` | Classification: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`. Regression: `r2`, `rmse`, `mae` |
| `cv_folds` (`cv`) | 0 | 0 or at least 2 |
| `class_weight` | none | `balanced`, `none` |
| `test_size` | 0.2 | between 0 and 1 |
| `random_state` (`seed`) | 14 | integer |
| `datetime_columns` | none | list of columns |
| `max_onehot` | 20 | integer |
| `plots` | `false` | `true`/`false` |
| `explain` | `none` | `none`, `permutation`, `shap` |
| `report` | `false` | `true` also turns plots on |
| `artifact_dir`, `output_dir` | `artifact`, `Output` | Relative folders inside the working directory |
| `versioned_runs`, `run_name` | `false` | Per-run sub folders |

## Example

```yaml
data: stud.csv
target: math_score
problem: Regression
features: auto
feature_selection: [correlation, importance]
models: [Linear Regression, Random Forest, HistGradientBoosting Regressor]
metric: rmse
cv_folds: 5
plots: true
report: true
```
