
<img src="logo.png" width="150" height="100">

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

To run this tool, you will require the following:

Download all the required packages

```bash
pip install -r requirements.txt

```

## Installation

Suggestion: Use a Virtual env for this to successfully run it 

1. Install This Application with pip install

```bash
  pip install JustDataML

```
2. Install by cloning this repository

```bash
  git clone https://github.com/jaideepsinhdabhi/JustDataML.git
```
Go into the Repo Folder

```bash
  cd JustDataML
```
Download all the required packages
```bash
  pip install -r requirements.txt
```
## Usage/Examples

```bash
python JDML.py --Config <Configfile> --Train --Predict <test.csv> --Output <Output Name>

```
### Arguments

- `-C, --Config`: You need to provide a configuration file to obtain all the necessary arguments for ML models to operate.

    Note: Make sure you give proper argument mentioned in the Document.

- `-T, --Train`: [Optional] If provided, it initiates model training based on the specified data_config file. 

    Note: Ensure that the Data_Config.csv file and Datafile are available in the "Data" folder.

- `-P, --Predict`: [Optional] If provided, performs prediction using the trained model. Make sure not to delete or modify the artifact folder to get the results.

- `-O, --Output`: [Required with -P (--Predict)] Specifies the output name for the predicted data frame generated from the test data.
    Note: it will also generate a Model_summary file and if Hyperparameter is given Yes in the Config file then it will generate a stats file for that too





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


    #####   A sample csv file and some Data are there in Data Folder for a demo Run

    #### Note: It will also generate logs into a logs folder for every run please check for every Run to get more idea on that.

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
