# Titanic Kaggle Project - Ensemble & Hyperparameter Tuning

## Overview
This folder contains two main notebooks addressing the Titanic survival prediction problem:

- **voting_classifier_titanic.ipynb**  
  Implements a Voting Classifier ensemble combining multiple classifiers.  
  Attempts to improve performance by aggregating base model predictions.

- **titanic_hyperparameter_tuning.ipynb**  
  Uses `RandomizedSearchCV` for hyperparameter tuning.  
  Aims to optimize the performance of a `RandomForestClassifier`.

- **titanic_survival_prediction_feature_importance.pt.1.ipynb**  
  Contains detailed feature engineering, preprocessing, and initial model evaluation steps.



## Key Learning Points

- Understand overfitting: high training accuracy but lower cross-validation accuracy indicates poor generalization.  
- Recognize the importance of hyperparameter tuning and feature engineering to mitigate overfitting.  
- Plan to explore techniques like max_depth limitation, feature selection, and alternative models to improve performance.

---

## How to Run
- Install required packages:
   pip install pandas numpy scikit-learn  

