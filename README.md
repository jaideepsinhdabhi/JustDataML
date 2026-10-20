
<img src="https://raw.githubusercontent.com/jaideepsinhdabhi/JustDataML/main/logo.png" width="150" height="100">

# JustDataML: Simplified Machine Learning for Everyone

Welcome to JustDataML, a user-friendly tool designed to make machine learning accessible to all, regardless of technical background. JustDataML automates the process of model selection, data preprocessing, and prediction, allowing you to focus on insights rather than coding complexities.

## Authors

- [@jaideepsinhdabhi](https://www.github.com/jaideepsinhdabhi)

## Features

- Automated Model Selection
- Streamlined Preprocessing
- Quick Predictions
- Flexible Configuration
- Hyperparameter Tuning
- Accurate Prediction

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
JDML --Config <Configfile> --Train --Predict <test.csv> --Output <Output Name>
```

Try the bundled demo:

```bash
cd examples
JDML -C Data_Config_clf.csv -T -P Data/bezdekIris.data -O demo
```

### Arguments

- `-C, --Config`: [Required with `-T` or `-P`] Path to the configuration file (see below).

- `-T, --Train`: Trains models based on the config file. The data file named in the config must be inside the `Data` folder.

- `-P, --Predict`: Predicts the rows of the given CSV with the trained model. The CSV must contain every column listed in `Features_Cols`. Keep the `artifact` folder from training next to where you run the command.

- `-O, --Output`: [Optional, only with `-P`] Name of the predictions file, saved as `Output/<name>.csv` (default `Target_outFile`). It must be a plain name; paths that leave the `Output` folder (such as `../x` or absolute paths) are rejected.

You can combine `-T` and `-P` to train and predict in one run. Training also writes `Output/Model_Summary.csv`, and `Output/Hyperparameter_stats.csv` when hyperparameter tuning is `Yes`.

### How the best model is chosen

Models are compared on a validation split taken from the training data (80/20 of the 80% training part). The best one is then refitted on all training data and scored once on the untouched test set; that is the "Best Model Test Score" printed at the end. Because the columns in `Model_Summary.csv` show validation scores for the comparison, they are named `*_validation`.

### Use it as a Python library

```python
import pandas as pd
from jdml.cli import Just_Data_ML
from jdml.Configuration import DataProcessor

config = DataProcessor("Data_Config.csv")   # same config file as the CLI
config.load_config()
config.read_data()                          # reads Data/<Data_Name>

jdml = Just_Data_ML()
jdml.Data_Train(
    Features_Cols=config.Features_Cols,
    Data_df=config.Data_df,
    Target_Col=config.Target_Col,
    Models=config.Models,
    Problem_Objective=config.Problem_Objective,
    HyperParamter_Yes_or_No=config.HyperParamter_Yes_or_No,
    Normalization_tech=config.scaler,
)

new_rows = pd.read_csv("new_data.csv")
result = jdml.Predict_test(config.Problem_Objective, config.Features_Cols, new_rows)
print(result["Target_Out"])                 # your DataFrame plus a Target_Out column
```

Files are written relative to the current working directory (`artifact/`, `Output/`). The library does not configure logging; call `jdml.Logger.setup_logging()` if you want a log file.

### Security notes

- Trained models are stored with `joblib` in `artifact/`, with a `.sha256` checksum next to each file. A model whose checksum is missing or does not match is refused. The checksum detects tampering and corruption; it cannot protect against someone who can rewrite both files, so only run predictions from an `artifact` folder you trust.
- Output and data file names from the command line and the config file cannot point outside the `Output` and `Data` folders.

### Upgrading from 0.0.x

Version 0.1.0 reorganises the package. Check these if you used an earlier release:

- The Python package is now `jdml` (all lower case) instead of `JDML`: `from JDML.JDML import Just_Data_ML` becomes `from jdml.cli import Just_Data_ML`, and `JDML.source.*` modules are now `jdml.*` (for example `jdml.Configuration`). The `JDML` command is unchanged.
- Python 3.10+ is required.
- Models trained with an earlier version are not loaded (no checksum files, renamed `preprocessor.pkl`). Train again.
- `Ridge Classification` is the model name; the old misspelling `Ridge Classifcaiton` is still accepted in config files.
- Bad configuration values (unknown model, wrong objective, target listed among the features, ...) now stop with an error message instead of printing a note and failing later.
- The sample data and configs moved to the `examples` folder.

### Development and releases

```bash
uv sync                 # install
uv run pytest --cov     # tests
uv run ruff check .     # lint
uv build                # build sdist + wheel into dist/
```

GitHub Actions runs the tests on every push and pull request. The package version comes from the git tag, so there is no version number to edit. To release, publish a GitHub Release with the tag you want, for example `v0.1.0` (the leading `v` is optional and removed). The release workflow builds that version, checks the built version equals the tag, runs the tests and publishes to PyPI. Releases marked as pre-release (for example tag `v0.2.0rc1`) go to TestPyPI instead. The tag must be a valid [PEP 440](https://peps.python.org/pep-0440/) version and higher than the one already on PyPI. Builds made without a tag get a development version such as `0.0.1.dev24+g1234abc`.

## Data Configuration File Example

This is an example of a data configuration file (`Data_Config.csv`) used with the JustData_ML (JDML) tool. This file specifies the necessary information for training and predicting with machine learning models.

### CSV Structure:

The CSV file contains the following fields:

- **Data_Name:** Name of the dataset file with extension and it should be present in the Data Folder (`bezdekIris.data` in this example).
- **Features_Cols:** Comma-separated list of feature columns in the dataset (`sepal length, sepal width, petal length, petal width` in this example).
- **Target_Col:** Name of the target column in the dataset (`class` in this example).
- **Problem_Objective:** Objective of the machine learning task (`Classification` or `Regression`).
- **Normalization_tech:** Normalization technique to be applied (`StandardScaler`, `MinMaxScaler`, etc.).
- **Model_to_include:** Models to include in the training process (`ALL` or specific models). (below Listed for Models)
- **HyperParamter_Yes_or_No:** Indicates whether hyperparameter tuning should be performed (`Yes` or `No`).


    #####   Sample config files and data are in the `examples` folder for a demo run (`Data_Config_clf.csv`, `Data_Config_reg.csv`).

    #### Note: The command line tool writes a log file into a `logs` folder for every run.

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
