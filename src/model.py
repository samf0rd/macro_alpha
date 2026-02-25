"""
Macro-Alpha Forecast Engine - Model Training & MLflow Tracking
==============================================================
Trains the XGBoost model, logs parameters and metrics to MLflow,
and saves the production artifacts.
"""

import pandas as pd
import joblib
import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_with_mlflow():
    logger.info("Loading model-ready features...")
    
    # 1. Load the data
    train_path = Path('C:/Users/Sam Garcia/PycharmProjects/macro_alpha/data/processed/train_ready_features.parquet')
    if not train_path.exists():
        logger.error("Data not found. Run data_pipeline.py and notebooks first.")
        return
        
    df_train = pd.read_parquet(train_path)
    
    X_train = df_train.drop(columns=['target_5d_up'])
    y_train = df_train['target_5d_up']
    feature_names = list(X_train.columns)
    
    # Define hyperparameters
    params = {
        "tree_method": "hist",
        "enable_categorical": True,
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1
    }

    # Set up MLflow Experiment
    mlflow.set_experiment("Macro_Alpha_Forecast")

    with mlflow.start_run(run_name="Baseline_XGBoost"):
        logger.info("Training Master Model...")
        
        # 2. Initialize and Train Model
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # 3. Calculate basic training metrics (In-sample just for tracking)
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_prec = precision_score(y_train, train_preds)
        
        # 4. Log everything to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_precision", train_prec)
        
        # Log the actual model to MLflow's artifact store
        mlflow.xgboost.log_model(model, "xgboost_model")
        
        logger.info(f"MLflow Run completed. Acc: {train_acc:.4f}, Prec: {train_prec:.4f}")

        # 5. Export Local Artifacts for Streamlit (Dashboard usage)
        models_dir = Path('C:/Users/Sam Garcia/PycharmProjects/macro_alpha/models')
        models_dir.mkdir(exist_ok=True)
        
        joblib.dump(model, models_dir / 'macro_xgb_model.joblib')
        joblib.dump(feature_names, models_dir / 'model_features.joblib')
        logger.info("✅ Local artifacts exported to /models/")

if __name__ == "__main__":
    train_with_mlflow()