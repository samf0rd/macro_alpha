import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent

print("🔍 Loading data for LSTM Feature Extraction...")
df = pd.read_parquet(PROJECT_ROOT / 'data' / 'processed' / 'train_ready_features.parquet')

# 1. Prepare Data for LSTM
features = ['daily_return', 'volatility_20d', 'RSI_14']
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

SEQUENCE_LENGTH = 10

# 🧠 2. Define the LSTM Architecture
class MarketLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(MarketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_day_out = lstm_out[:, -1, :]
        out = self.fc(last_day_out)
        return self.sigmoid(out)

model = MarketLSTM(input_size=len(features), hidden_size=32, num_layers=2)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 3. Create Sequences for the ENTIRE dataset
X_all, y_all = [], []
for i in range(len(scaled_features) - SEQUENCE_LENGTH):
    X_all.append(scaled_features[i:(i + SEQUENCE_LENGTH)])
    y_all.append(df['target_5d_up'].iloc[i + SEQUENCE_LENGTH])

X_all_tensor = torch.FloatTensor(np.array(X_all))
y_all_tensor = torch.FloatTensor(np.array(y_all)).unsqueeze(1)

print(f"🚀 Training LSTM on all {len(X_all)} historical windows...")
# Train on the whole dataset to extract the best historical embeddings
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_all_tensor)
    loss = criterion(outputs, y_all_tensor)
    loss.backward()
    optimizer.step()

# 4. Generate the Momentum Score for the dataset
print("📊 Generating LSTM Momentum Scores...")
model.eval()
with torch.no_grad():
    lstm_scores = model(X_all_tensor).numpy().flatten()

# 5. Append the scores to the dataframe
# Note: The first 10 days of the dataframe get dropped because they don't have enough history
df_ensemble = df.iloc[SEQUENCE_LENGTH:].copy()
df_ensemble['lstm_momentum_score'] = lstm_scores

# Save the Ultimate Master Dataset
out_path = PROJECT_ROOT / 'data' / 'processed' / 'train_ready_features_ensemble.parquet'
df_ensemble.to_parquet(out_path)

print(f"✅ Ensemble dataset saved to {out_path} with {len(df_ensemble)} rows.")
print("New Feature 'lstm_momentum_score' successfully injected!")