# Titanic Analytics — Module 2

## Overview

This module performs exploratory data analysis (EDA) and predictive modeling
using the Titanic dataset.

The pipeline includes:

- Dataset loading and offline storage
- Data inspection
- Missing-value analysis
- Missing-value handling
- Univariate analysis
- Bivariate analysis
- Correlation analysis
- Data visualization
- Feature standardization
- Logistic Regression
- Random Forest
- Model evaluation
- Confusion matrices
- ROC curves
- Feature importance
- Model comparison

---

## Dataset

The Titanic dataset contains 891 passenger records and 15 columns.

The offline dataset is stored at:

```text
analytics/titanic.csv

## Detailed EDA Interpretations

### Survival by Sex
Female passengers had a substantially higher observed survival rate than male passengers. This shows a clear association between sex and survival in this dataset.

### Survival by Passenger Class
First-class passengers had the highest observed survival rate, followed by second and third class. Passenger class therefore shows a clear association with survival in this dataset.

### Survival by Sex and Passenger Class
Combining sex and passenger class reveals patterns that are less visible when considering either variable separately. Female passengers in first and second class had particularly high observed survival rates, while male passengers in second and third class had much lower rates.

### Fare by Survival
Fare distributions differ between passengers who survived and those who did not. Higher fares are associated with passenger groups showing higher observed survival, although this is an association rather than a causal conclusion.

### Age by Survival
Age distributions differ between the two survival groups. This visualization helps examine age patterns among survivors and non-survivors.

## Classification and Imbalance

The classifiers are evaluated using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Results are saved in `model_comparison.csv`.

The imbalance experiment compares baseline Logistic Regression, class-weight balancing, and SMOTE applied only to the training data. Precision, recall, and F1-score are compared across these approaches.

## Regression

The regression side task predicts `fare` from other available Titanic features. MAE, RMSE, R², and Adjusted R² are reported, together with a residual plot and heteroscedasticity diagnostic.

## Model Persistence

The complete fitted best pipeline is saved as:

`analytics/models/best_pipeline.joblib`

It contains preprocessing and the final estimator and is reloaded with `joblib.load()` for a raw-input prediction test.

## Run

```bash
python analytics/titanic_analysis.py
python analytics/modeling.py
```
