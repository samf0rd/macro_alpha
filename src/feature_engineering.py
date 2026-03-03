"""
Macro-Alpha Forecast Engine - Feature Engineering
===============================================
Transforms raw market/macro data into stationary ML features.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def engineer_features():
    logger.info("Loading raw data for feature engineering...")
    data_path = PROJECT_ROOT / 'data' / 'market_macro_data.parquet'
    df = pd.read_parquet(data_path)
    
    # 1. Technical Features
    df.ta.rsi(close='close_sp500', length=14, append=True)
    df.ta.macd(close='close_sp500', fast=12, slow=26, signal=9, append=True)
    df['daily_return'] = df['close_sp500'].pct_change()
    df['volatility_20d'] = df['daily_return'].rolling(window=20).std() * np.sqrt(252)
    
    # 2. Stationary Moving Average Ratios
    sma_50 = df['close_sp500'].rolling(50).mean()
    sma_200 = df['close_sp500'].rolling(200).mean()
    df['price_to_sma_50'] = df['close_sp500'] / sma_50
    df['price_to_sma_200'] = df['close_sp500'] / sma_200
    df['golden_cross'] = (sma_50 > sma_200).astype(int)
    
    # 3. Macro Velocity Lags
    df['yield_spread_1mo_change'] = df['yield_spread'].diff(21)
    df['fed_funds_3mo_change'] = df['fed_funds_rate'].diff(63)
    df['yield_10y_change'] = df['yield_10y'].diff()
    df['yield_2y_change'] = df['yield_2y'].diff()
    df['yield_spread_change'] = df['yield_spread'].diff()
    df['fed_funds_6mo_lag'] = df['fed_funds_rate'].shift(126)

    # 4. Market Regimes (HMM)
    logger.info("Calculating 3-State Market Regimes...")
    # Create a temporary dataframe without NaNs just to train the HMM
    hmm_data = df[['daily_return', 'volatility_20d']].dropna()
    
    hmm_model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000, random_state=42)
    hmm_model.fit(hmm_data)
    
    # Predict the states and align them with the main dataframe
    df.loc[hmm_data.index, 'regime'] = hmm_model.predict(hmm_data)
    
    # 5. Target Variable
    df['future_5d_close'] = df['close_sp500'].shift(-5)
    df['target_5d_up'] = (df['future_5d_close'] > df['close_sp500']).astype(int)
    
# 6. Clean & Export
    columns_to_drop = [
        'close_sp500', 'cpi', 'future_5d_close', 
        'yield_10y', 'yield_2y', 'yield_spread'
    ]
    
    # --- TRAINING DATA ---
    # We drop the last 5 days because they don't have a known target yet, then drop NaNs
    df_train = df[:-5].dropna()
    df_train_ready = df_train.drop(columns=columns_to_drop)
    
    # --- INFERENCE DATA ---
    # We drop the future target columns FIRST, so we don't accidentally delete "Today" when dropping NaNs
    df_inference_ready = df.drop(columns=columns_to_drop).dropna()
    
    out_dir = PROJECT_ROOT / 'data' / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = out_dir / 'train_ready_features.parquet'
    inference_path = out_dir / 'inference_ready_features.parquet'
    
    df_train_ready.to_parquet(train_path)
    df_inference_ready.to_parquet(inference_path)
    
    logger.info("✅ Feature engineering complete.")
    logger.info(f"   Saved {len(df_train_ready)} training rows to {train_path}")
    logger.info(f"   Saved {len(df_inference_ready)} inference rows to {inference_path}")

if __name__ == "__main__":
    engineer_features()