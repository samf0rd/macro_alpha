"""
Macro-Alpha Forecast Engine - Price Action Expert (LSTM)
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

class MarketLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MarketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)

def train_price_action_expert():
    logger.info("Loading Price Action Data...")
    df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'train_ready_features.parquet')
    
    features = ['daily_return', 'volatility_20d', 'RSI_14']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[features])
    
    # Save the scaler so the Streamlit app can use it live!
    models_dir = PROJECT_ROOT / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / 'lstm_scaler.joblib')
    
    SEQUENCE_LENGTH = 10
    X_seq, y_seq = [], []
    for i in range(len(scaled_features) - SEQUENCE_LENGTH):
        X_seq.append(scaled_features[i:(i + SEQUENCE_LENGTH)])
        y_seq.append(df['target_5d_up'].iloc[i + SEQUENCE_LENGTH])

    X_train = torch.FloatTensor(np.array(X_seq))
    y_train = torch.FloatTensor(np.array(y_seq)).unsqueeze(1)
    
    logger.info("Training LSTM Neural Network...")
    model = MarketLSTM(input_size=len(features), hidden_size=32, num_layers=2)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
    torch.save(model.state_dict(), models_dir / 'lstm_model.pt')
    logger.info("✅ Price Action Expert saved to models/lstm_model.pt")

if __name__ == "__main__":
    train_price_action_expert()