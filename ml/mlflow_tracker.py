# This file handles MLflow experiment tracking for ML models

import mlflow  # mlflow tracking library
import mlflow.sklearn  # mlflow sklearn integration for logging sklearn models
from dotenv import load_dotenv  # load env variables
import os  # access env variables

load_dotenv()  # load .env file

def init_mlflow():
    # set mlflow tracking uri where runs will be stored
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    # set experiment name — creates new if not exists
    mlflow.set_experiment("shoppers_classification")
    print("MLflow initialized")  # log init

def log_run(model, model_name, acc, pre, rec, f1, auc, is_best=False):
    # log a single model run to mlflow
    with mlflow.start_run(run_name=model_name):  # start a new mlflow run
        # log model hyperparameters
        mlflow.log_param("model_type", model_name)  # log model name
        mlflow.log_param("n_estimators", 100)  # log number of trees
        mlflow.log_param("is_best_model", is_best)  # log if this is best model

        # log evaluation metrics
        mlflow.log_metric("accuracy", acc)    # log accuracy
        mlflow.log_metric("precision", pre)   # log precision
        mlflow.log_metric("recall", rec)      # log recall
        mlflow.log_metric("f1_score", f1)     # log f1 score
        mlflow.log_metric("roc_auc", auc)     # log roc auc

        # log model artifact
        mlflow.sklearn.log_model(model, model_name)  # save model in mlflow
        print(f"MLflow run logged for {model_name} — Best: {is_best}")  # log success

def log_all_runs(rf_model, rf_metrics, xgb_model, xgb_metrics, best_name):
    # initialize mlflow once
    init_mlflow()

    # log random forest run
    log_run(
        rf_model, "RandomForest",
        rf_metrics['acc'], rf_metrics['pre'],
        rf_metrics['rec'], rf_metrics['f1'], rf_metrics['auc'],
        is_best=(best_name == "RandomForest")  # mark if best
    )

    # log xgboost run
    log_run(
        xgb_model, "XGBoost",
        xgb_metrics['acc'], xgb_metrics['pre'],
        xgb_metrics['rec'], xgb_metrics['f1'], xgb_metrics['auc'],
        is_best=(best_name == "XGBoost")  # mark if best
    )