# EMI Prediction AI
---
## Overview
- A dual-model Machine Learning project that combines classification and regression techniques to predict both the EMI eligibility and the maximum EMI amount a candidate can avail.
- Predict whether a candidate is **eligible for an EMI** (classification).
- Estimate the **maximum EMI amount** that the candidate can avail (regression).
---
## Objectives 
- Automate EMI Eligibility prediction based on candidate profile.
- Predict the maximum EMI Amount for eligible candidates.
- Compare and evaluate multiple ML models for performance and reliability.
- Track experiments, models and metrics using **MLFlow**.
---
## Tech Stack
- Language : Python 3.10
- Libraries : pandas,matplotlib,seaborn,scikit-learn,xgboost,mlflow.
- Tools : Jupyter notebook, MLFlow tracking & Model Registry.
---
## Dataset 
- Contains applicant details and financial metrics used for EMI eligibility analysis.

| Variable | Description |
|-----------|-------------|
| monthly_salary | Monthly income of the applicant |
| years_of_employment | Total years of employment |
| monthly_rent | Rent amount paid per month |
| family_size | Number of family members |
| dependents | Number of dependents financially supported |
| credit_score | Credit score of applicant |
| bank_balance | Current account balance |
| current_emi_amount | Existing EMI obligations |
| requested_amount | EMI amount requested |
| requested_tenure | Tenure (in months) of requested EMI |
| emi_eligible | Target variable for classification (0/1) |
| max_emi_amount | Target variable for regression (numeric) |
---
## Methodology
### 1. Data Preprocessing
   - Handles missing values, duplicates, inconsistencies.
   - Encoded categorical features and scaled numeric features for Linear regression and Logistic regression.
   - Split dataset into train-validation-test split.
   - Engineered new ratios and features like expenses-income ratio, debt-income ratio, etc.
### 2. EDA (Exploratory Data Analysis)
   - Visualized correlations between income, expenses, and EMI eligibility.
   - Analyzed distributions and feature importance.
   - Analyzed demographics characterictics with emi eligibility.
### 3. Model Building
#### ***Regression models (Maximum EMI Amount)***
   - Linear Regression
   - Random Forest
   - XGBoost
   - Gradient Boosting
#### ***Classification Models (EMI Eligibility)***
   - Logistic Regression
   - Random Forest
   - XGBoost
### 4. Model Evaluation 
   - Classification Metrics - Accuracy, Precision, Recall, F1_score
   - Regression Metrics - MAE, RMSE, R2_score
### 5. MLFlow Experiment Tracking
   - Tracked model parameters, metrics and artifacts using ***mlflow tracking server***
   - Registered best performing models in ***mlflow model registry***
   - Compared experiment runs and model performance for model selection.
---
## Results
#### Regression 
|Model|RMSE|MAE|R2_score|
|-----|---|----|--------|
|Linear Regression|4147.68|2969.54|0.715|
|Random Forest|1039.60|426.82|0.982|
|XGBoost|792.59|393.69|0.989|
|Gradient Boosting|1459.38|800.76|0.964|
|XGBoost test_split|783.34|	392.41|	0.989|
|Gradient Boosting on test	|1439.36|	797.25|	0.965|

#### Classification
|Model|Accuracy|Precision|Recall|F1_score|
|-----|--------|---------|------|--------|
|Logistic Regression|0.785596|	0.648205	|0.733686	|0.629772|
|Random Forest|0.883899	|0.686989	|0.760025	|0.703918|	
|XGBoost|0.964387|	0.899490	|0.762274	|0.796113	|
|Random Forest (test split)|0.884448	|0.685603	|0.761751	|0.703158	|
|XGBoost (test split)|0.963746	|0.884130	|0.751122	|0.781796|	
---
## Requirements
```bash
python == 3.10
pandas
numpy
scikit-learn
matplotlib
seaborn
mlflow
jupyter
xgboost
```
---
## Author
#### **Nilesh Bahirgaonkar**
Data Scientist 
- gmail - nileshbahirgaonkar1494@gmail.com
-  [LinkedIn Profile](https://www.linkedin.com/in/nilesh-bahirgaonkar/)
