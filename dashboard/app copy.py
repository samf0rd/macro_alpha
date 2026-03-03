import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas_ta as ta
from pathlib import Path

# --- Dynamic Path Resolution ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- LSTM Class Definition ---
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

# --- Page Config ---
st.set_page_config(page_title="Macro-Alpha Engine", layout="wide", initial_sidebar_state="expanded")

# --- Feature Dictionary ---
FEATURE_NAMES = {
    'vix': 'VIX (Fear Gauge)',
    'fed_funds_rate': 'Federal Funds Rate',
    'RSI_14': 'RSI (Momentum)',
    'MACD_12_26_9': 'MACD (Trend)',
    'MACDh_12_26_9': 'MACD Histogram',
    'MACDs_12_26_9': 'MACD Signal',
    'daily_return': 'Daily S&P 500 Return',
    'volatility_20d': '20-Day Volatility',
    'price_to_sma_50': 'Price vs 50-Day SMA',
    'price_to_sma_200': 'Price vs 200-Day SMA',
    'golden_cross': 'Golden Cross Signal',
    'yield_spread_1mo_change': 'Yield Spread Chg (1M)',
    'fed_funds_3mo_change': 'Fed Funds Chg (3M)',
    'yield_10y_change': '10Y Yield Daily Chg',
    'yield_2y_change': '2Y Yield Daily Chg',
    'yield_spread_change': 'Yield Curve Spread Chg',
    'fed_funds_6mo_lag': 'Fed Funds Rate (6mo Lag)',
    'regime': 'HMM Market Regime'
}

# --- Load Artifacts ---
@st.cache_resource
def load_data():
    model_path = PROJECT_ROOT / 'models' / 'macro_xgb_model.joblib'
    model = joblib.load(model_path)
    
    lstm_scaler = joblib.load(PROJECT_ROOT / 'models' / 'lstm_scaler.joblib')
    lstm_model = MarketLSTM(input_size=3, hidden_size=32, num_layers=2)
    lstm_model.load_state_dict(torch.load(PROJECT_ROOT / 'models' / 'lstm_model.pt', map_location=torch.device('cpu'), weights_only=True))
    lstm_model.eval()
    
    features_path = PROJECT_ROOT / 'data' / 'processed' / 'inference_ready_features.parquet'
    df_features = pd.read_parquet(features_path)
    
    expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else model.get_booster().feature_names
    X = df_features[[col for col in expected_features if col in df_features.columns]]
    
    y = df_features['target_5d_up'] if 'target_5d_up' in df_features.columns else None
    
    raw_path = PROJECT_ROOT / 'data' / 'market_macro_data.parquet'
    alt_raw_path = PROJECT_ROOT / 'src' / 'data' / 'market_macro_data.parquet'
    
    if raw_path.exists():
        df_raw = pd.read_parquet(raw_path)
    elif alt_raw_path.exists():
        df_raw = pd.read_parquet(alt_raw_path)
    else:
        df_raw = df_features.copy()
        
    df_raw = df_raw.ffill().dropna()
    return model, lstm_model, lstm_scaler, X, y, df_raw, df_features

model, lstm_model, lstm_scaler, X_inference, y_inference, df_raw, df_features = load_data()

latest_date = X_inference.index[-1].date()
X_today = X_inference.iloc[-1:]
raw_today = df_raw.iloc[-1:]

# --- Sidebar ---
st.sidebar.title("🚀 Macro-Alpha Engine")
st.sidebar.caption(f"Status: Online | Data: {latest_date}")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Risk Settings")
conf_thresh = st.sidebar.slider("Conviction Threshold", 0.50, 0.75, 0.55, 0.01)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", ["📈 Daily Prediction Desk", "📊 Portfolio Performance", "🧪 Scenario Lab & History"])

# --- INFERENCE: The Double Brain Logic ---
prob_up = model.predict_proba(X_today)[0][1]

last_10_days = df_features[['daily_return', 'volatility_20d', 'RSI_14']].iloc[-10:]
scaled_10_days = lstm_scaler.transform(last_10_days)
tensor_10_days = torch.FloatTensor(np.array([scaled_10_days]))
with torch.no_grad():
    lstm_prob = lstm_model(tensor_10_days).item()

macro_vote = prob_up > conf_thresh
lstm_vote = lstm_prob > conf_thresh
ensemble_vote = macro_vote and lstm_vote

if ensemble_vote:
    sig_txt, direction = "BULLISH", "UP"
    box_bg, box_border, text_color = "#d1e7dd", "#badbcc", "#0f5132" 
elif not macro_vote and not lstm_vote:
    sig_txt, direction = "BEARISH", "DOWN"
    box_bg, box_border, text_color = "#f8d7da", "#f5c2c7", "#842029" 
else:
    sig_txt, direction = "NEUTRAL (MIXED SIGNALS)", "FLAT"
    box_bg, box_border, text_color = "#fff3cd", "#ffecb5", "#664d03" 


# ==========================================
# PAGE 1: DAILY PREDICTION DESK
# ==========================================
if page == "📈 Daily Prediction Desk":
    
    with st.container(border=True):
        st.subheader("Market Context & Technicals")
        if 'close_sp500' in df_raw.columns:
            chart_data = df_raw.iloc[-252:].copy()
            chart_data['SMA_50'] = chart_data['close_sp500'].rolling(50).mean()
            chart_data.ta.rsi(close='close_sp500', length=14, append=True)
            chart_data.ta.macd(close='close_sp500', fast=12, slow=26, signal=9, append=True)
            
            # Look at the last 126 days for a clean chart
            view_data = chart_data.iloc[-126:]

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])

            # Changed S&P 500 to bright white and removed 200 SMA
            fig.add_trace(go.Scatter(x=view_data.index, y=view_data['close_sp500'], name='S&P 500', line=dict(color='#FFFFFF', width=2.5)), row=1, col=1)
            fig.add_trace(go.Scatter(x=view_data.index, y=view_data['SMA_50'], name='50 SMA', line=dict(color='#3b82f6', dash='dot')), row=1, col=1)

            if 'RSI_14' in view_data.columns:
                fig.add_trace(go.Scatter(x=view_data.index, y=view_data['RSI_14'], name='RSI (14)', line=dict(color='#8b5cf6')), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="#10b981", row=2, col=1)

            if 'MACDh_12_26_9' in view_data.columns:
                hist_colors = np.where(view_data['MACDh_12_26_9'] >= 0, '#10b981', '#ef4444')
                fig.add_trace(go.Bar(x=view_data.index, y=view_data['MACDh_12_26_9'], name='Histogram', marker_color=hist_colors), row=3, col=1)
                fig.add_trace(go.Scatter(x=view_data.index, y=view_data['MACD_12_26_9'], name='MACD', line=dict(color='#60a5fa')), row=3, col=1)
                fig.add_trace(go.Scatter(x=view_data.index, y=view_data['MACDs_12_26_9'], name='Signal', line=dict(color='#f59e0b')), row=3, col=1)

            fig.update_layout(height=550, margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        with st.container(border=True):
            st.subheader("Today's Signal")
            
            st.markdown(f"""
            <div style="background-color: {box_bg}; border: 1px solid {box_border}; padding: 25px; border-radius: 8px; text-align: center; margin-bottom: 20px;">
                <h1 style="color: {text_color}; margin: 0; font-size: 2.5rem; font-weight: 800;">{sig_txt}</h1>
                <p style="color: {text_color}; margin: 5px 0 0 0; font-weight: 600; font-size: 1.1rem;">Expected 5-Day Move: {direction}</p>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Macro Conf.", f"{prob_up:.1%}")
            c2.metric("LSTM Conf.", f"{lstm_prob:.1%}")
            c3.metric("Threshold", f"{conf_thresh:.0%}")
        
    with col2:
        with st.container(border=True):
            st.subheader("🤖 AI Analyst Summary")
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_today)
            shap_df = pd.DataFrame({'Feature': X_today.columns, 'Impact': shap_values[0].values})
            shap_df['Name'] = shap_df['Feature'].map(FEATURE_NAMES).fillna(shap_df['Feature'])
            
            bullish = shap_df[shap_df['Impact'] > 0].sort_values('Impact', ascending=False).head(3)
            bearish = shap_df[shap_df['Impact'] < 0].sort_values('Impact', ascending=True).head(3)
            
            if ensemble_vote:
                st.success(f"**Edge Identified: Bullish.** Both the Macro and LSTM brains are aligned. The primary tailwind is **{bullish.iloc[0]['Name']}**, supported by favorable short-term momentum.")
            elif not macro_vote and not lstm_vote:
                st.error(f"**Edge Identified: Bearish.** Downward pressure is being driven by macroeconomic headwinds like **{bearish.iloc[0]['Name']}**, and confirmed by negative price momentum.")
            else:
                st.info(f"**Edge Identified: Neutral (Cash).** The Dual-Brain engine is detecting mixed signals between short-term price action and long-term macro fundamentals. Defaulting to capital preservation.")
                
            with st.expander("View Top Macro Drivers Breakdown", expanded=True):
                cb1, cb2 = st.columns(2)
                with cb1:
                    st.markdown("**Bullish Factors:**")
                    for _, row in bullish.iterrows(): 
                        val = X_today[row['Feature']].iloc[0]
                        st.markdown(f"🟢 **{row['Name']}** ({val:.4f})")
                with cb2:
                    st.markdown("**Bearish Factors:**")
                    for _, row in bearish.iterrows(): 
                        val = X_today[row['Feature']].iloc[0]
                        st.markdown(f"🔴 **{row['Name']}** ({val:.4f})")

# ==========================================
# PAGE 2: PORTFOLIO PERFORMANCE
# ==========================================
elif page == "📊 Portfolio Performance":
    
    with st.container(border=True):
        st.title("🧠 Unsupervised Market Regimes")
        st.warning("⚠️ **Analyst Note:** This chart visualizes our Hidden Markov Model mapping. True *Out-of-Sample* performance is tracked via MLflow.")
        
        lookback = min(750, len(X_inference))
        plot_df = df_features.iloc[-lookback:].copy()
        
        if 'regime' not in plot_df.columns: plot_df['regime'] = 0 
            
        if 'close_sp500' in df_raw.columns:
            plot_df['close'] = df_raw.loc[plot_df.index, 'close_sp500']
        else:
            plot_df['close'] = plot_df['daily_return'].cumsum()

        fig = go.Figure()
        regime_colors = {0: '#10b981', 1: '#f59e0b', 2: '#ef4444'} 
        regime_names = {0: 'Quiet Bull (Low Vol)', 1: 'Choppy/Transition (Med Vol)', 2: 'Extreme Bear (High Vol)'}
        
        for regime_id in [0, 1, 2]:
            mask = plot_df['regime'] == regime_id
            fig.add_trace(go.Scatter(
                x=plot_df[mask].index, y=plot_df[mask]['close'], mode='markers',
                marker=dict(color=regime_colors.get(regime_id, 'gray'), size=4),
                name=regime_names.get(regime_id, f'Regime {regime_id}')
            ))

        fig.update_layout(height=400, hovermode="x unified", margin=dict(l=0, r=0, t=10, b=0), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        
        # Calculate Advanced ML Metrics
        X_backtest = X_inference.iloc[-lookback:]
        historical_probs = model.predict_proba(X_backtest)[:, 1]
        
        # Signals: 1 (Buy) or 0 (Cash)
        signals = np.where(historical_probs > conf_thresh, 1, 0)
        actual_returns = plot_df['daily_return'].shift(-1).fillna(0)
        
        # 1. Precision (When the model says BUY, how often is it right?)
        true_positives = np.sum((signals == 1) & (actual_returns > 0))
        precision = true_positives / np.sum(signals) if np.sum(signals) > 0 else 0
        
        # 2. Overall Accuracy (Base model probability vs Actual market direction)
        raw_binary_preds = np.where(historical_probs > 0.50, 1, 0)
        actual_binary_dir = np.where(actual_returns > 0, 1, 0)
        overall_accuracy = np.mean(raw_binary_preds == actual_binary_dir)
        
        st.divider()
        st.markdown("##### 🧪 Model Diagnostics (Historical)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Long Precision (Win Rate)", f"{precision:.1%}", help="When the model issues a BUY signal, what percentage of the time does the market actually go up over the next 5 days?")
        c2.metric("Base Model Accuracy", f"{overall_accuracy:.1%}", help="Overall classification accuracy of the macro model.")
        c3.metric("Time in Market", f"{np.sum(signals)} / {lookback} Days", help="Number of days the model was fully invested vs sitting in defensive cash.")
        
        st.divider()
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Strategy Sharpe", f"{strat_sharpe:.2f}")
        c2.metric("Max Drawdown", f"{max_dd:.2%}")
        c3.metric("Long Precision (Win Rate)", f"{precision:.1%}")
        c4.metric("Base Model Accuracy", f"{overall_accuracy:.1%}")
        c5.metric("Time in Market", f"{np.sum(signals)} / {lookback} Days")
    
    with st.container(border=True):
        st.subheader("🎛️ Current Macro Environment")
        mc1, mc2, mc3, mc4 = st.columns(4)
        
        val_sp500 = raw_today.get('close_sp500', pd.Series([np.nan])).iloc[0]
        val_vix = raw_today.get('vix', pd.Series([np.nan])).iloc[0]
        val_spread = raw_today.get('yield_spread', pd.Series([np.nan])).iloc[0]
        val_fed = raw_today.get('fed_funds_rate', pd.Series([np.nan])).iloc[0]
        
        mc1.metric("S&P 500 Price", f"{val_sp500:.2f}" if pd.notna(val_sp500) else "N/A")
        mc2.metric("VIX Level", f"{val_vix:.2f}" if pd.notna(val_vix) else "N/A")
        mc3.metric("10Y-2Y Spread", f"{val_spread:.2f}%" if pd.notna(val_spread) else "N/A")
        mc4.metric("Effective Fed Funds", f"{val_fed:.2f}%" if pd.notna(val_fed) else "N/A")

# ==========================================
# PAGE 3: SCENARIO LAB & HISTORY
# ==========================================
elif page == "🧪 Scenario Lab & History":
    
    with st.container(border=True):
        st.title("🧪 What-If Scenario Lab")
        col_sim, col_res = st.columns([1, 1])
        
        with col_sim:
            st.markdown("**Adjust Inputs:**")
            sim_vix = st.slider("VIX (Fear Gauge)", 10.0, 85.0, float(X_today['vix'].iloc[0]), 0.5)
            sim_fed = st.slider("Fed Funds Rate (%)", 0.0, 8.0, float(X_today['fed_funds_rate'].iloc[0]), 0.25)
            sim_sma = st.slider("Price to 200-SMA Ratio", 0.70, 1.30, float(X_today['price_to_sma_200'].iloc[0]), 0.01)
            sim_yield_chg = st.slider("10Y Yield Daily Chg (bps)", -0.50, 0.50, float(X_today['yield_10y_change'].iloc[0]), 0.01)
            
            X_sim = X_today.copy()
            X_sim['vix'] = sim_vix
            X_sim['fed_funds_rate'] = sim_fed
            X_sim['price_to_sma_200'] = sim_sma
            X_sim['yield_10y_change'] = sim_yield_chg

        with col_res:
            st.markdown("**Simulated Outcome:**")
            sim_prob = model.predict_proba(X_sim)[0][1]
            if sim_prob > conf_thresh:
                s_txt, s_bg, s_border, s_txt_col = "BULLISH", "#d1e7dd", "#badbcc", "#0f5132"
            elif sim_prob < (1 - conf_thresh):
                s_txt, s_bg, s_border, s_txt_col = "BEARISH", "#f8d7da", "#f5c2c7", "#842029"
            else:
                s_txt, s_bg, s_border, s_txt_col = "NEUTRAL", "#fff3cd", "#ffecb5", "#664d03"
                
            st.markdown(f"""
            <div style="background-color: {s_bg}; border: 1px solid {s_border}; padding: 15px; border-radius: 8px; text-align: center; margin-bottom: 15px;">
                <h2 style="color: {s_txt_col}; margin: 0; font-weight: 800;">{s_txt}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Simulated Macro Confidence", f"{sim_prob:.1%}", delta=f"{(sim_prob - prob_up)*100:.1f}% change")
            
            st.divider()
            alerts = 0
            if sim_vix > 35:
                st.error("⚠️ **Volatility:** High VIX suggests panic selling.")
                alerts += 1
            if sim_fed > 6.0 and sim_prob < 0.50:
                st.warning("⚠️ **Monetary Tightening:** High rates suppressing equities.")
                alerts += 1
            if sim_sma < 0.90:
                st.error("📉 **Trend Breakdown:** Price significantly below long-term support.")
                alerts += 1
            if alerts == 0:
                st.success("✅ No critical macro stress-fractures detected.")

    with st.container(border=True):
        st.subheader("📜 Prediction History (Last 14 Trading Days)")
        if y_inference is not None:
            hist_X = X_inference.iloc[-14:].copy()
            hist_y = y_inference.iloc[-14:]
            hist_probs = model.predict_proba(hist_X)[:, 1]
            
            hist_data = []
            for i in range(len(hist_X)):
                p = hist_probs[i]
                if p > conf_thresh: 
                    sig = "🟢 BUY"
                elif p < (1-conf_thresh): 
                    sig = "🔴 SELL"
                else: 
                    sig = "🟡 CASH"
                
                actual = hist_y.iloc[i]
                outcome = "UP" if actual == 1 else "DOWN" if actual == 0 else "Unknown"
                
                # Grading the Model
                if outcome == "Unknown":
                    result = "⏳ Pending"
                elif sig == "🟡 CASH":
                    result = "➖ Neutral"
                elif (sig == "🟢 BUY" and outcome == "UP") or (sig == "🔴 SELL" and outcome == "DOWN"):
                    result = "✅ Correct"
                else:
                    result = "❌ Incorrect"
                    
                hist_data.append([hist_X.index[i].date(), f"{p:.1%}", sig, outcome, result])
                
            hist_df = pd.DataFrame(hist_data, columns=['Date', 'Confidence', 'Signal', 'Actual Move', 'Result'])
            st.dataframe(hist_df.iloc[::-1], hide_index=True, use_container_width=True)