import pandas as pd
import numpy as np
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load the Model-Ready Data
df = pd.read_parquet('data/processed/train_ready_features.parquet')             
X = df.drop(columns=['target_5d_up'])
y = df['target_5d_up']

def objective(trial):
    # 2. Define the Hyperparameter Search Space
    params = {
        'tree_method': 'hist',
        'enable_categorical': True,
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'random_state': 42,
        'n_jobs': -1
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    precisions = []
    
    # 3. Walk-Forward Validation inside the trial
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict using a strict 55% confidence threshold (matching your strategy)
        probs = model.predict_proba(X_test)[:, 1]
        preds = np.where(probs > 0.55, 1, 0)
        
        # If the model never predicted 1 (too conservative), precision is 0
        if sum(preds) == 0:
            prec = 0.0
        else:
            prec = precision_score(y_test, preds)
            
        precisions.append(prec)
        
    # We want Optuna to MAXIMIZE the average precision across all time folds
    return np.mean(precisions)

# 4. Run the Optimization
print("🚀 Starting Optuna Hyperparameter Tuning...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50) # Runs 50 different combinations

print("\n=========================================")
print(f"🏆 Best Average Precision: {study.best_value:.2%}")
print("Optimal Parameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")
print("=========================================")