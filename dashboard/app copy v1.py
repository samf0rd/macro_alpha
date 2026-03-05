import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas_ta as ta
from sklearn.metrics import confusion_matrix, f1_score
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
    'vix': 'VIX (Fear Gauge)', 'fed_funds_rate': 'Federal Funds Rate', 'RSI_14': 'RSI (Momentum)',
    'MACD_12_26_9': 'MACD (Trend)', 'MACDh_12_26_9': 'MACD Histogram', 'MACDs_12_26_9': 'MACD Signal',
    'daily_return': 'Daily S&P 500 Return', 'volatility_20d': '20-Day Volatility',
    'price_to_sma_50': 'Price vs 50-Day SMA', 'price_to_sma_200': 'Price vs 200-Day SMA',
    'golden_cross': 'Golden Cross Signal', 'yield_spread_1mo_change': 'Yield Spread Chg (1M)',
    'fed_funds_3mo_change': 'Fed Funds Chg (3M)', 'yield_10y_change': '10Y Yield Daily Chg',
    'yield_2y_change': '2Y Yield Daily Chg', 'yield_spread_change': 'Yield Curve Spread Chg',
    'fed_funds_6mo_lag': 'Fed Funds Rate (6mo Lag)', 'regime': 'HMM Market Regime'
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
page = st.sidebar.radio("Navigation", ["📈 Daily Prediction Desk", "📊 Portfolio Performance", "🧪 Scenario Lab & History", "📖 Methodology & Architecture"])

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
    sig_txt, direction = "NEUTRAL (MIXED)", "FLAT"
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
            view_data = chart_data.iloc[-126:]

            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.2, 0.2])

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
                        
# Phase A: Advanced SHAP Waterfall UI Compromise
    with st.expander("📊 View Advanced SHAP Waterfall Analysis", expanded=False):
        st.markdown("This mathematical breakdown shows exactly how each feature pushed the model's probability away from the baseline.")
        
        X_display = X_today.rename(columns=FEATURE_NAMES)
        shap_explainer = shap.Explainer(model)
        shap_obj = shap_explainer(X_display)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        shap.plots.waterfall(shap_obj[0], max_display=8, show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

# ==========================================
# PAGE 2: PORTFOLIO PERFORMANCE
# ==========================================
elif page == "📊 Portfolio Performance":
    
    with st.container(border=True):
        st.title("🧠 Unsupervised Market Regimes")
        st.info("💡 **Methodology:** We use a Hidden Markov Model (HMM) to mathematically cluster the S&P 500 into three distinct volatility regimes. This prevents our models from applying 'bull market rules' during a crash.")
        
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
        
        # Strict ML Evaluation Metrics
        X_backtest = X_inference.iloc[-lookback:]
        historical_probs = model.predict_proba(X_backtest)[:, 1]
        
        raw_binary_preds = np.where(historical_probs > conf_thresh, 1, 0)
        actual_returns = plot_df['daily_return'].shift(-1).fillna(0)
        actual_binary_dir = np.where(actual_returns > 0, 1, 0)
        
        precision = np.sum((raw_binary_preds == 1) & (actual_binary_dir == 1)) / np.sum(raw_binary_preds) if np.sum(raw_binary_preds) > 0 else 0
        overall_accuracy = np.mean(raw_binary_preds == actual_binary_dir)
        model_f1 = f1_score(actual_binary_dir, raw_binary_preds)
        
        st.divider()
        st.markdown("##### 🧪 Model Diagnostics (Classification Performance)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Long Precision", f"{precision:.1%}", help="When the model votes BUY, how often does the market actually go up?")
        c2.metric("Overall Accuracy", f"{overall_accuracy:.1%}", help="Total correct predictions (Buys and Cash/Sells) vs total days.")
        c3.metric("F1-Score", f"{model_f1:.3f}", help="Harmonic mean of precision and recall. A better metric than accuracy for imbalanced financial data.")
        c4.metric("Time in Market", f"{np.sum(raw_binary_preds)} / {lookback} Days", help="Number of days the model was fully invested vs sitting in defensive cash.")
        
        st.markdown("###### Confusion Matrix")
        cm = confusion_matrix(actual_binary_dir, raw_binary_preds)
        fig_cm = ff.create_annotated_heatmap(
            z=cm, 
            x=['Predicted CASH', 'Predicted BUY'], 
            y=['Actual DOWN', 'Actual UP'], 
            colorscale='Blues',
            showscale=False
        )
        fig_cm.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with st.container(border=True):
        st.subheader("🎛️ Current Macro Environment")
        mc1, mc2, mc3, mc4 = st.columns(4)
        
        val_sp500 = raw_today.get('close_sp500', pd.Series([np.nan])).iloc[0]
        val_vix = raw_today.get('vix', pd.Series([np.nan])).iloc[0]
        val_spread = raw_today.get('yield_spread', pd.Series([np.nan])).iloc[0]
        val_fed = raw_today.get('fed_funds_rate', pd.Series([np.nan])).iloc[0]
        
        mc1.metric("S&P 500 Price", f"{val_sp500:.2f}" if pd.notna(val_sp500) else "N/A")
        mc2.metric("VIX Level", f"{val_vix:.2f}", help="CBOE Volatility Index. Values > 30 indicate extreme market fear.")
        mc3.metric("10Y-2Y Spread", f"{val_spread:.2f}%", help="Yield Curve Spread. Negative values represent an inverted yield curve, a classic recession indicator.")
        mc4.metric("Effective Fed Funds", f"{val_fed:.2f}%", help="The baseline interest rate set by the Federal Reserve.")

# ==========================================
# PAGE 3: SCENARIO LAB & HISTORY
# ==========================================
elif page == "🧪 Scenario Lab & History":
    
    with st.container(border=True):
        st.title("🧪 What-If Scenario Lab")
        
        # Phase A: Quick Scenarios
        scenario = st.radio("⚡ Quick Scenarios", ["Current Market", "Fed Hikes 50bps", "Market Panic (VIX 40+)", "Strong Bull Trend"], horizontal=True)
        
        def_vix = float(X_today['vix'].iloc[0])
        def_fed = float(X_today['fed_funds_rate'].iloc[0])
        def_sma = float(X_today['price_to_sma_200'].iloc[0])
        def_y10 = float(X_today['yield_10y_change'].iloc[0])
        
        if scenario == "Fed Hikes 50bps":
            def_fed += 0.50
        elif scenario == "Market Panic (VIX 40+)":
            def_vix = max(40.0, def_vix)
            def_sma = min(0.85, def_sma)
        elif scenario == "Strong Bull Trend":
            def_vix = min(15.0, def_vix)
            def_sma = max(1.10, def_sma)

        col_sim, col_res = st.columns([1, 1])
        with col_sim:
            st.markdown("**Adjust Inputs:**")
            sim_vix = st.slider("VIX (Fear Gauge)", 10.0, 85.0, def_vix, 0.5)
            sim_fed = st.slider("Fed Funds Rate (%)", 0.0, 8.0, def_fed, 0.25)
            sim_sma = st.slider("Price to 200-SMA Ratio", 0.70, 1.30, def_sma, 0.01)
            sim_yield_chg = st.slider("10Y Yield Daily Chg (bps)", -0.50, 0.50, def_y10, 0.01)
            
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
                outcome = "UP" if actual == 1 else "DOWN" if actual == 0 else "Pending"
                
                # Accountabilty P/L
                target_date = hist_X.index[i]
                try:
                    current_price = df_raw.loc[target_date, 'close_sp500']
                    future_idx = df_raw.index.get_loc(target_date) + 5
                    if future_idx < len(df_raw):
                        future_price = df_raw['close_sp500'].iloc[future_idx]
                        p_l = (future_price - current_price) / current_price
                        pl_str = f"+{p_l:.2%}" if p_l > 0 else f"{p_l:.2%}"
                    else:
                        pl_str = "TBD"
                except:
                    pl_str = "TBD"
                
                # Grading the Model
                if outcome == "Pending": result = "⏳ Pending"
                elif sig == "🟡 CASH": result = "➖ Neutral"
                elif (sig == "🟢 BUY" and outcome == "UP") or (sig == "🔴 SELL" and outcome == "DOWN"): result = "✅ Correct"
                else: result = "❌ Incorrect"
                    
                hist_data.append([hist_X.index[i].date(), f"{p:.1%}", sig, outcome, pl_str, result])
                
            hist_df = pd.DataFrame(hist_data, columns=['Date', 'Confidence', 'Signal', 'Actual Move', '5-Day P/L', 'Result'])
            st.dataframe(hist_df.iloc[::-1], hide_index=True, use_container_width=True)

# ==========================================
# PAGE 4: METHODOLOGY & ARCHITECTURE
# ==========================================
elif page == "📖 Methodology & Architecture":
    st.title("📖 System Architecture & Methodology")
    
    st.markdown("""
    This application is powered by a **Dual-Brain Parallel Voting Ensemble**. 
    Rather than relying on a single monolithic model, the engine separates market forecasting into two distinct domains: Fundamental Macroeconomics and Short-Term Price Action.
    """)
    
    st.divider()
    
    col_mac, col_lstm = st.columns(2)
    
    with col_mac:
        st.markdown("### 🌳 Brain 1: The Risk Manager (XGBoost)")
        st.markdown("""
        **Domain:** Macroeconomic Fundamentals & Broad Market Context.
        
        This model asks: *"Are the underlying economic conditions safe for investing right now?"* It evaluates a wide array of slow-moving, structural features, including:
        * **The Yield Curve:** 10Y-2Y Spread and short-term debt velocities.
        * **Monetary Policy:** Effective Federal Funds Rate.
        * **Market Fear:** VIX (Volatility Index).
        * **Trend Baselines:** Price deviations against the 50-day and 200-day Simple Moving Averages.
        
        Using **SHAP (SHapley Additive exPlanations)**, the model exposes its feature importance dynamically, offering complete transparency into its decision-making process.
        """)
        
    with col_lstm:
        st.markdown("### 🧠 Brain 2: The Momentum Trader (LSTM)")
        st.markdown("""
        **Domain:** Short-Term Price Action & Volatility.
        
        This model asks: *"Regardless of the macro economy, is there profitable short-term momentum in the chart right now?"*
        
        Built with **PyTorch**, this Long Short-Term Memory (LSTM) neural network completely ignores interest rates and the economy. Instead, it looks exclusively at the sequence of the last 10 trading days:
        * **Daily S&P 500 Returns**
        * **Rolling 20-Day Volatility**
        * **RSI (Relative Strength Index)**
        
        By analyzing sequences, it captures complex temporal patterns that standard tree-based models miss.
        """)

    st.divider()
    
    st.markdown("### 🤝 The Handshake: Parallel Voting Logic")
    st.markdown("""
    To generate a final **BULLISH / BUY** signal, the engine requires consensus. Both the XGBoost Macro Brain and the PyTorch LSTM Brain must independently output a confidence score that exceeds the user-defined `Conviction Threshold`. 
    
    * If the Macro environment is favorable but short-term momentum is negative, the model protects capital and votes **CASH**.
    * If short-term momentum is surging but the yield curve and VIX signal impending danger, the model ignores the trap and votes **CASH**.
    * Only when structural fundamentals align with immediate price momentum does the model aggressively vote **LONG**.
    """)
    
    st.divider()
    
    st.markdown("### 🤖 Unsupervised Market Regimes (HMM)")
    st.markdown("""
    Financial markets are non-stationary; the rules of a quiet bull market simply do not apply during a crash. 
    
    To solve this, the pipeline incorporates an **Unsupervised Hidden Markov Model (HMM)**. Before the XGBoost model makes a prediction, the HMM clusters the current market data into one of three distinct mathematical states (Regimes) based on latent volatility and return profiles. This allows the primary forecasting model to dynamically adjust its learned rules based on the immediate market context, significantly reducing catastrophic drawdowns during transitionary periods.
    """)