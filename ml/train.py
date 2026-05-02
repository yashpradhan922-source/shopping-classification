# This file trains ML models and logs results to MLflow

import pandas as pd  # data manipulation
from sklearn.model_selection import train_test_split  # split data into train and test sets
from sklearn.ensemble import RandomForestClassifier  # random forest model
from xgboost import XGBClassifier  # xgboost model
from sklearn.metrics import (
    accuracy_score,    # overall accuracy
    precision_score,   # precision score
    recall_score,      # recall score
    f1_score,          # f1 score
    roc_auc_score      # roc auc score
)
import pickle  # save model to disk
from dotenv import load_dotenv  # load env variables
import os  # access env variables
from preprocess import load_transformed_data, preprocess  # import preprocess functions
from mlflow_tracker import log_all_runs  # import mlflow logging function

load_dotenv()  # load .env file

def train():
    df = load_transformed_data()  # load transformed data from postgresql
    X, y = preprocess(df)  # apply scaling, feature selection and SMOTE

    # split into 80% train and 20% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train random forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # fit on training data
    print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_model.predict(X_test)):.4f}")

    # train xgboost model
    xgb_model = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)  # fit on training data
    print(f"XGBoost Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.4f}")

    # calculate metrics for random forest
    rf_metrics = {
        'acc': accuracy_score(y_test, rf_model.predict(X_test)),          # accuracy
        'pre': precision_score(y_test, rf_model.predict(X_test)),          # precision
        'rec': recall_score(y_test, rf_model.predict(X_test)),             # recall
        'f1':  f1_score(y_test, rf_model.predict(X_test)),                 # f1 score
        'auc': roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])  # roc auc
    }

    # calculate metrics for xgboost
    xgb_metrics = {
        'acc': accuracy_score(y_test, xgb_model.predict(X_test)),          # accuracy
        'pre': precision_score(y_test, xgb_model.predict(X_test)),          # precision
        'rec': recall_score(y_test, xgb_model.predict(X_test)),             # recall
        'f1':  f1_score(y_test, xgb_model.predict(X_test)),                 # f1 score
        'auc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:,1])  # roc auc
    }

    # select best model based on accuracy
    if xgb_metrics['acc'] >= rf_metrics['acc']:
        best_model = xgb_model  # xgboost better or equal
        best_name = "XGBoost"
    else:
        best_model = rf_model  # random forest better
        best_name = "RandomForest"

    print(f"Best Model: {best_name}")  # log best model name

    # log both models to mlflow for comparison
    log_all_runs(rf_model, rf_metrics, xgb_model, xgb_metrics, best_name)

    # save best model to disk
    os.makedirs('./ml', exist_ok=True)  # create ml dir if not exists
    with open(os.getenv('MODEL_PATH'), 'wb') as f:
        pickle.dump(best_model, f)  # serialize model
    print(f"Model saved to {os.getenv('MODEL_PATH')}")  # log save path

    return best_model, X_test, y_test, best_name  # return for evaluate use

if __name__ == "__main__":
    train()  # run training