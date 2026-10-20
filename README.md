
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jaideepsinhdabhi/JustDataML/main/JDML-dark.png">
  <img src="https://raw.githubusercontent.com/jaideepsinhdabhi/JustDataML/main/JDML.png" width="300" alt="JDML">
</picture>

# JustDataML: Simplified Machine Learning for Everyone

Welcome to JustDataML, a user-friendly tool designed to make machine learning accessible to all, regardless of technical background. JustDataML automates the process of model selection, data preprocessing, and prediction, allowing you to focus on insights rather than coding complexities.

## Authors

- [@jaideepsinhdabhi](https://www.github.com/jaideepsinhdabhi)

## Features

- Automated model selection across scikit-learn and XGBoost models (LightGBM and CatBoost optional)
- **Automatic feature selection**: leave `features: auto` and JDML drops IDs, constant and redundant columns and keeps the useful ones
- **YAML or JSON config** (the old CSV config still works but is deprecated)
- **Plots and an HTML report**: model comparison, feature importance, confusion matrix, ROC, residuals and more
- Choice of metric (accuracy, F1, ROC AUC, R2, RMSE, ...) and cross-validation scores
- Missing values, dates and high-cardinality text columns handled for you
- Hyperparameter tuning, class-imbalance handling, versioned runs, `metrics.json` for every run
- A Python API for notebooks: `jdml.fit(df, target="y")`

## Requirements

- Python 3.10 or newer
- On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`

## Installation

Use a virtual environment.

1. From PyPI:

```bash
pip install JustDataML
# or, with uv
uv pip install JustDataML

# optional extras
pip install "JustDataML[plots]"      # charts and the HTML report (matplotlib)
pip install "JustDataML[lightgbm]"   # LightGBM models
pip install "JustDataML[catboost]"   # CatBoost models
pip install "JustDataML[shap]"       # SHAP explanations
pip install "JustDataML[all]"        # everything above
```

2. From source (development), using [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/jaideepsinhdabhi/JustDataML.git
cd JustDataML
uv sync            # creates .venv and installs the package with dev tools
uv run pytest      # run the tests
```

## Usage/Examples

The command is `JDML` (installed with the package). Run it from a working folder that contains a `Data/` folder with your data file; the tool creates `artifact/`, `Output/` and `logs/` next to it.

```bash
JDML init                              # writes a template config.yaml
JDML train   -C config.yaml            # train, pick the best model, write reports
JDML predict -C config.yaml new.csv -O result
```

The older flag form still works: `JDML -C config.yaml -T -P new.csv -O result`.

Try the bundled demo:

```bash
cd examples
JDML train -C config_clf.yaml
JDML predict -C config_clf.yaml Data/bezdekIris.data -O demo
```

### A minimal config

```yaml
data: bezdekIris.data        # file inside the Data folder
target: class
problem: Classification      # or Regression
features: auto               # JDML chooses the useful columns (or list them)
plots: true
```

Everything else has a default. The full list of settings is in the table below and in [docs/configuration.md](docs/configuration.md).

### Commands and flags

- `JDML init [-o file] [--type classification|regression] [--force]`: write a template config.
- `JDML train -C <config>` / `-T`: train using the config. The data file must be inside the `Data` folder.
- `JDML predict -C <config> <test.csv> [-O name]` / `-P <test.csv>`: predict the rows of the CSV. The CSV needs the feature columns JDML selected during training (listed in `artifact/features.json`); other columns are ignored and kept in the output.
- `-O, --Output`: name of the predictions file, saved as `Output/<name>.csv` (default `Target_outFile`). It must be a plain name; paths that leave the output folder are rejected.

You can combine `-T` and `-P` to train and predict in one run.

### Config settings

| Setting | Default | Meaning |
|---|---|---|
| `data` | required | Data file inside `Data/` (CSV text) |
| `target` | required | Column to predict |
| `problem` | required | `Classification` or `Regression` |
| `features` | `auto` | `auto` or a list of columns |
| `feature_selection` | none (`importance` when `features: auto`) | One or more of `none`, `variance`, `correlation`, `importance`, `mutual_info`, `rfe` |
| `top_k` | 0 (automatic) | Most features to keep for ranking methods |
| `correlation_threshold` | 0.95 | `correlation` drops a column this correlated with one already kept |
| `models` | `ALL` | `ALL` (core models) or a list of names |
| `scaler` | `StandardScaler` | `StandardScaler`, `MinMaxScaler`, `RobustScaler`, `MaxAbsScaler` |
| `tuning` | `false` | Grid search hyperparameters |
| `metric` | `accuracy` / `r2` | Classification: `accuracy`, `f1`, `precision`, `recall`, `roc_auc`. Regression: `r2`, `rmse`, `mae` |
| `cv_folds` | 0 | 2 or more adds cross-validation mean and std to the summary |
| `class_weight` | none | `balanced` for imbalanced classes (models that support it) |
| `test_size`, `random_state` | 0.2, 14 | Train/test split |
| `datetime_columns` | none | Columns to turn into year, month, day, weekday, day-of-year |
| `max_onehot` | 20 | Text columns with more values are target encoded instead of one-hot |
| `plots` | `false` | Save charts to `Output/plots/` (needs `JustDataML[plots]`) |
| `explain` | `none` | `permutation` or `shap` importance plot |
| `report` | `false` | Also write `Output/report.html` |
| `artifact_dir`, `output_dir` | `artifact`, `Output` | Folders inside the working directory |
| `versioned_runs`, `run_name` | `false` | Keep each run in its own sub folder (`run_name` or a timestamp); prediction uses the latest run |

### Automatic feature selection

With `features: auto`, JDML takes every column except the target, removes constant columns and ID-like columns (text or whole-number columns where every row is unique), and then ranks the rest with a random forest and keeps the ones that cover 95% of the importance. Choose other steps with `feature_selection`, for example `[correlation, importance]` to drop duplicates first. Selection only sees the training rows, so test data never influences it. The result is in `Output/selected_features.csv` (score and the reason each column was dropped) and `artifact/features.json`; prediction uses the same columns, so you do not list features again.

### Outputs

| File | Content |
|---|---|
| `Output/Model_Summary.csv` | Scores of every model (train and validation), plus cross-validation columns |
| `Output/metrics.json` | Best model, scores, features, versions, seed |
| `Output/selected_features.csv` | Feature selection report |
| `Output/Hyperparameter_stats.csv` | Grid search results (when `tuning: true`) |
| `Output/plots/*.png` | Charts (when `plots: true`) |
| `Output/report.html` | Everything in one page (when `report: true`) |
| `artifact/` | Trained model, preprocessor, label encoder, features list, each with a `.sha256` file |

Plots: model comparison, feature importance, feature selection scores, confusion matrix, ROC and precision-recall curves (classification), predicted vs actual and residuals (regression), target distribution, correlation heatmap and a learning curve.

### How the best model is chosen

Models are compared on a validation split taken from the training data (80/20 of the training part). The best one is then refitted on all training data and scored once on the untouched test set; that is the "Best Model Test ..." score printed at the end. Because the columns in `Model_Summary.csv` show validation scores for the comparison, they are named `*_validation`. For error metrics (`rmse`, `mae`) the lowest value wins.

### Use it as a Python library

```python
import pandas as pd
import jdml

df = pd.read_csv("Data/stud.csv")
model = jdml.fit(df, target="math_score", feature_selection="importance", plots=True)

print(model)             # Model(Regression, best='Gradient Boosting', r2=0.87, features=5)
model.report()           # DataFrame with one row per model tried
model.features           # the columns JDML kept
model.predict(new_rows)  # Series with the same index as new_rows
```

`jdml.fit` takes the same settings as the YAML file (`problem` is guessed from the target when omitted). Files are written to `artifact/` and `Output/` (change with `artifact_dir` and `output_dir`). Reopen a trained model later with `jdml.Model.load("Regression")`.

The lower-level classes (`DataProcessor`, `Just_Data_ML`) are still available:

```python
from jdml.cli import Just_Data_ML
from jdml.Configuration import DataProcessor

config = DataProcessor("config.yaml")
config.load_config()
config.read_data()
Just_Data_ML().Data_Train(
    Features_Cols=config.Features_Cols, Data_df=config.Data_df, Target_Col=config.Target_Col,
    Models=config.Models, Problem_Objective=config.Problem_Objective,
    HyperParamter_Yes_or_No=config.HyperParamter_Yes_or_No, Normalization_tech=config.scaler,
    options=config.options, paths=config.base_paths)
```

The library does not configure logging; call `jdml.Logger.setup_logging()` if you want a log file.

### Security notes

- Trained models are stored with `joblib` in `artifact/`, with a `.sha256` checksum next to each file. A model whose checksum is missing or does not match is refused. The checksum detects tampering and corruption; it cannot protect against someone who can rewrite both files, so only run predictions from an `artifact` folder you trust.
- Output and data file names from the command line and the config file cannot point outside the `Output` and `Data` folders.

### Upgrading

**From 0.1.x to 0.2**

- Config files can now be YAML or JSON (`JDML init` writes a template). CSV configs still work and show a deprecation note; the old names (`Data_Name`, `Features_Cols`, ...) are accepted in YAML and JSON too.
- `features` can be `auto`. Models trained by 0.2 save the chosen feature list, so prediction no longer needs one.
- `ALL` now includes HistGradientBoosting (12 core regression models and 11 classification models). LightGBM and CatBoost models are only used when you name them.
- `Data_Train` returns a result dict and `Predict_test` takes an optional `paths` argument. Retrain models made with 0.1.x.

**From 0.0.x**

- The Python package is now `jdml` (all lower case) instead of `JDML`: `from JDML.JDML import Just_Data_ML` becomes `from jdml.cli import Just_Data_ML`, and `JDML.source.*` modules are now `jdml.*` (for example `jdml.Configuration`). The `JDML` command is unchanged.
- Python 3.10+ is required.
- Models trained with an earlier version are not loaded (no checksum files, renamed `preprocessor.pkl`). Train again.
- `Ridge Classification` is the model name; the old misspelling `Ridge Classifcaiton` is still accepted in config files.
- Bad configuration values (unknown model, wrong objective, target listed among the features, ...) now stop with an error message instead of printing a note and failing later.

### Development and releases

```bash
uv sync                 # install
uv run pytest --cov     # tests
uv run ruff check .     # lint
uv build                # build sdist + wheel into dist/
```

GitHub Actions runs the tests on every push and pull request. The package version comes from the git tag, so there is no version number to edit. To release, publish a GitHub Release with the tag you want, for example `v0.1.0` (the leading `v` is optional and removed). The release workflow builds that version, checks the built version equals the tag, runs the tests and publishes to PyPI. Releases marked as pre-release (for example tag `v0.2.0rc1`) go to TestPyPI instead. The tag must be a valid [PEP 440](https://peps.python.org/pep-0440/) version and higher than the one already on PyPI. Builds made without a tag get a development version such as `0.0.1.dev24+g1234abc`.

## Converting an old CSV config

The old CSV had one setting per row: `Data_Name,stud.csv` and `Features_Cols,a,b,c`. The same file as YAML:

```yaml
data: stud.csv
features: [a, b, c]
target: math_score
problem: Regression
scaler: RobustScaler
models: ALL
tuning: true      # was HyperParamter_Yes_or_No,Yes
```

Sample configs and data are in the `examples` folder (`config_clf.yaml`, `config_reg.yaml`, `config_reaction.json`). The command line tool writes a log file into a `logs` folder for every run.

## Available Models for Regression and Classification

### Regression Models:

1. **Random Forest**
   - Description: Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training and outputting the mean prediction of the individual trees.

2. **Decision Tree**
   - Description: Decision Tree is a non-parametric supervised learning method used for classification and regression. It works by partitioning the input space into regions and predicting the target variable based on the average of the training instances in the corresponding region.

3. **Gradient Boosting**
   - Description: Gradient Boosting is a machine learning technique for regression and classification problems that builds models in a stage-wise manner and tries to fit new models to the residuals of the previous models.

4. **Linear Regression**
   - Description: Linear Regression is a linear approach to modelling the relationship between a dependent variable and one or more independent variables.

5. **XGBRegressor**
   - Description: XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

6. **Support Vector Reg**
   - Description: Support Vector Regression (SVR) is a type of Support Vector Machine (SVM) algorithm that is used to predict a continuous variable.

7. **Linear Ridge**
   - Description: Ridge Regression is a linear regression technique that is used to analyze multiple regression data that suffer from multicollinearity.

8. **Linear Lasso**
   - Description: Lasso regression is a type of linear regression that uses shrinkage. It penalizes the absolute size of the regression coefficients.

9. **ElasticNet**
   - Description: ElasticNet is a linear regression model that combines the properties of Ridge Regression and Lasso Regression.

10. **AdaBoost Regressor**
    - Description: AdaBoost (Adaptive Boosting) is an ensemble learning method that combines multiple weak learners to create a strong learner.

11. **KNeighborsRegressor**
    - Description: KNeighborsRegressor is a simple, non-parametric method used for regression tasks based on the k-nearest neighbours algorithm.

12. **HistGradientBoosting Regressor**
    - Description: A fast histogram-based gradient boosting model from scikit-learn that handles large datasets well.

Optional (install the extra and name them in `models`): **LightGBM Regressor** (`JustDataML[lightgbm]`) and **CatBoost Regressor** (`JustDataML[catboost]`).

### Classification Models:

1. **Logistic Regression**
   - Description: Logistic Regression is a statistical method for analyzing a dataset in which there are one or more independent variables that determine an outcome.

2. **Ridge Classification**
   - Description: Ridge Classifier is a classifier that uses Ridge Regression to classify data points.

3. **GaussianNB**
   - Description: Gaussian Naive Bayes is a simple probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.

4. **KNeighborsClassifier**
   - Description: KNeighborsClassifier is a simple, instance-based learning algorithm used for classification tasks based on the k-nearest neighbours algorithm.

5. **Decision Tree Classifier**
   - Description: Decision Tree Classifier is a non-parametric supervised learning method used for classification.

6. **Random Forest Classifier**
   - Description: Random Forest Classifier is an ensemble learning method for classification that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes (classification) of the individual trees.

7. **Support Vector Classifier**
   - Description: Support Vector Classifier (SVC) is a type of Support Vector Machine (SVM) algorithm that is used for classification tasks.

8. **AdaBoost Classifier**
   - Description: AdaBoost Classifier is an ensemble learning method that combines multiple weak learners to create a strong learner.

9. **Gradient Boosting Classifier**
   - Description: Gradient Boosting Classifier is a machine learning technique for classification problems that builds models in a stage-wise manner and tries to fit new models to the residuals of the previous models.

10. **XGBClassifier**
    - Description: XGBoost Classifier is an optimized distributed gradient boosting library designed for classification problems.

11. **HistGradientBoosting Classifier**
    - Description: A fast histogram-based gradient boosting classifier from scikit-learn.

Optional (install the extra and name them in `models`): **LightGBM Classifier** (`JustDataML[lightgbm]`) and **CatBoost Classifier** (`JustDataML[catboost]`).

These are the available regression and classification models supported by the JDML tool. You can use them for training and prediction based on your specific machine-learning tasks.

## Feedback

If you have any feedback or suggestions, please reach out to us at jaideep.dabhi7603@gmail.com


## Acknowledgements

 - [Krish Naik Playlist of End-to-End Machine Learning Project](https://youtube.com/playlist?list=PLZoTAELRMXVPS-dOaVbAux22vzqdgoGhG&si=Q1cXWDAZIdeeiC3m) This playlist really helped a lot I followed till the end and tried to code at the same time. 


 - [Krish Naik Github Repo for Above Project ](https://github.com/krishnaik06/mlproject) Github Repo from the playlist for reference. 
 - [How to Build a Complete Python Package Step-by-Step by ArjanCodes](https://youtu.be/5KEObONUkik?si=QYVU4lu8tpfmyTXV). This video really helped me to write a Python package and published it to the [PyPi](pypi.org)
 
# Hi, I'm Jaideepsinh Dabhi (jD)! 👋


## 🚀 About Me

🚀 Data Scientist | Analytics Enthusiast | Python Aficionado 🐍 \
I'm a Data Scientist working in the BioTech Industry.\
I am based in India 🇮🇳 \
I love to code in Python, Bash and R \
I have a strong base in statistics and Machine learning \
I am passionate about networking and fostering meaningful connections within the tech community. Feel free to reach out if you'd like to discuss Data Science 👨🏻‍💻, Machine Learning 🦾, Chess ♞ or Pens 🖋️


## 🔗 Links
[![GitHub](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/jaideepsinhdabhi)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jaideepsinh-d-543476167/)



## My Updates
👩‍💻 I'm currently working on Enhancing and trying to get more features into this Tool

🧠 I'm currently learning Gen AI and Computer Vision

👯‍♀️ I'm looking to collaborate on BioInformatics or Machine learning Front

💬 Ask me about Statistics and ML



## 🛠 Skills
Statistics, AWS/Cloud, Python Programming, Data Science, Machine Learning, Bioinformatics,  R Programming, SAS, Programming, SQL , Big Data Tools, Hadoop, PySpark.

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0) - see the [LICENSE](LICENSE) file for details.
