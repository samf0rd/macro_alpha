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

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def train_with_mlflow():
    logger.info("Loading model-ready features...")
    
    # 1. Load the data
    train_path = PROJECT_ROOT / 'data' / 'processed' / 'train_ready_features.parquet'
    if not train_path.exists():
        logger.error("Data not found. Run data_pipeline.py and notebooks first.")
        return
        
    df_train = pd.read_parquet(train_path)
    
    # Drop the target AND the LSTM score (XGBoost is purely the Macro/Regime expert now)
    cols_to_drop = ['target_5d_up', 'lstm_momentum_score']
    X_train = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    y_train = df_train['target_5d_up']
    feature_names = list(X_train.columns)
    
    # The mathematically optimal 61.16% Parameters (Macro + HMM Regime)
    params = {
        "n_estimators" : 170,
        "learning_rate": 0.0774,
        "max_depth": 4,
        "subsample": 0.5981,
        "colsample_bytree" : 0.5341,
        "min_child_weight" : 3,
        "gamma" : 0.0544,
        "tree_method": "hist",
        "enable_categorical" : True,
        "random_state" : 42,
        "n_jobs" : -1
    }

    # Set up MLflow Experiment
    mlflow.set_experiment("Macro_Alpha_Forecast")

    with mlflow.start_run(run_name="Parallel_Macro_Expert"):
        logger.info("Training Macro Expert Model...")
        
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
        models_dir = PROJECT_ROOT / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, models_dir / 'macro_xgb_model.joblib')
        joblib.dump(feature_names, models_dir / 'model_features.joblib')
        logger.info("✅ Local artifacts exported to /models/")

if __name__ == "__main__":
    train_with_mlflow()