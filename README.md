# 🚀 Macro-Alpha Forecast Engine

> An end-to-end Machine Learning pipeline that predicts S&P 500 directional movement using macroeconomic indicators and technical analysis.

**Author:** Samuel Garcia  
**Tech Stack:** Python | XGBoost | MLflow | Streamlit | Docker | AWS  
**Status:** 🏗️ In Development

---

## 📋 Project Overview

This is a **flagship capstone project** demonstrating production-grade ML engineering skills for quantitative finance. Unlike typical Kaggle competitions with pre-cleaned data, this project:

- ✅ Ingests **live data** from Yahoo Finance and FRED APIs
- ✅ Handles **real-world messiness** (missing data, business days, API failures)
- ✅ Implements **proper time-series validation** (no data leakage)
- ✅ Deploys with **MLOps best practices** (MLflow tracking, automated retraining)
- ✅ Serves predictions via **cloud-hosted dashboard** (Streamlit on AWS)

### The Business Problem

Can we predict short-term S&P 500 movements by combining:
- **Market signals:** Price momentum, volatility (VIX)
- **Macro indicators:** Interest rates, yield curve, inflation

**Target:** Binary classification—Will the S&P 500 close higher or lower 5 trading days from now?

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Data Sources   │
│  (yfinance +    │
│   FRED API)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ETL Pipeline  │
│  (data_pipeline │
│      .py)       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Feature      │
│  Engineering    │
│ (RSI, MACD,     │
│  lagged macro)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   XGBoost       │
│   Classifier    │
│ (Time-series    │
│  validation)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  MLflow         │
│  Experiment     │
│  Tracking       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Streamlit     │
│   Dashboard     │
│ (AWS EC2)       │
└─────────────────┘
```

---

## 📁 Project Structure

```
macro_alpha/
├── data/
│   ├── raw/                    # Original fetched data
│   └── processed/              # Clean, feature-engineered data
│
├── notebooks/                  # Exploratory analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_experiments.ipynb
│
├── src/                        # Production code
│   ├── data_pipeline.py        # ETL script
│   ├── feature_engineering.py  # Feature creation
│   ├── model.py                # Training & inference
│   └── utils.py                # Helper functions
│
├── models/                     # Saved model artifacts
├── mlflow/                     # Experiment tracking
├── dashboard/                  # Streamlit app
├── tests/                      # Unit tests
├── logs/                       # Execution logs
│
├── .github/workflows/          # CI/CD automation
│   └── daily_inference.yml     # Scheduled model runs
│
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone https://github.com/samf0rd/macro_alpha.git
cd macro_alpha

# Create virtual environment
py -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Fetch Data

```bash
cd src
python data_pipeline.py
```

This will:
- Download 15 years of S&P 500 + macro data
- Clean and merge on business days
- Save to `data/processed/market_macro_data.parquet`

### 3. Train Model (Coming Soon)

```bash
python model.py --mode train
```

### 4. Launch Dashboard (Coming Soon)

```bash
cd dashboard
streamlit run app.py
```

---

## 🧠 Technical Approach

### Data Sources

| Source | Indicators | Frequency |
|--------|-----------|-----------|
| **Yahoo Finance** | S&P 500 (^GSPC), VIX (^VIX) | Daily |
| **FRED** | 10Y/2Y yields, Fed Funds, CPI | Daily/Monthly |

### Feature Engineering

**Market Features:**
- Rolling volatility (10/30/60 day windows)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Price momentum (5/10/20 day returns)

**Macro Features:**
- Yield curve slope (10Y - 2Y spread)
- Real interest rates (Fed Funds - Inflation)
- Lagged macro variables (6-month lag for policy effects)

**Target Variable:**
- Binary: `1` if S&P 500 closes higher 5 days from now, `0` otherwise

### Model

- **Algorithm:** XGBoost Classifier
- **Validation:** Walk-forward time-series split (no shuffling!)
- **Evaluation Metrics:**
  - Precision (when model says "buy", how often is it right?)
  - Recall (what % of up-days does it catch?)
  - AUC-ROC (overall discrimination ability)
  - Risk-adjusted returns (Sharpe ratio vs. buy-and-hold)

### Key Design Decisions

**Why XGBoost?**
- Handles non-linear relationships in financial data
- Built-in feature importance
- Fast training on tabular data
- Proven track record in Kaggle finance competitions

**Why NOT LSTM?**
- XGBoost outperforms LSTM on tabular features with < 50 features
- LSTM requires more data and careful tuning
- (May add LSTM ensemble in Phase 2)

**Time-Series Split Logic:**
```
Train: 2010-2015 → Test: 2016
Train: 2010-2016 → Test: 2017
Train: 2010-2017 → Test: 2018
...
```
This simulates real trading: you can only use past data to predict the future.

---

## 🛡️ Preventing Common Pitfalls

### 1. Look-Ahead Bias ❌
**Problem:** Using tomorrow's data to predict today  
**Solution:** 
- Forward-fill macro data (use last known value)
- Never include future returns in training features
- Strict date alignment in merges

### 2. Non-Stationarity ❌
**Problem:** XGBoost can't extrapolate trends  
**Solution:** Convert prices → log returns (mean-reverting)

### 3. Data Leakage ❌
**Problem:** Target variable info leaking into features  
**Solution:** 
- Create features BEFORE splitting train/test
- Use only lagged macro variables
- Validate with walk-forward splits

---

## 📊 Results (Coming Soon)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | TBD |
| **Precision** | TBD |
| **Recall** | TBD |
| **AUC-ROC** | TBD |
| **Sharpe Ratio** | TBD |

### Example Prediction Output

```
Date: 2024-02-15
Prediction: 🟢 BULLISH (68% confidence)
Key Drivers:
  1. Yield spread widening (+25 bps) → Risk-on sentiment
  2. VIX declining (-3.2 pts) → Lower uncertainty
  3. Fed Funds rate stable → No tightening shock
```

---

## 🗓️ Development Roadmap

### ✅ Phase 1: Foundation (Weeks 1-2)
- [x] ETL pipeline with live data ingestion
- [ ] Feature engineering (RSI, MACD, lagged macro)
- [ ] Stationarity testing (ADF test)
- [ ] Walk-forward validation setup
- [ ] Baseline XGBoost model

### 🚧 Phase 2: MLOps (Weeks 3-4)
- [ ] MLflow experiment tracking
- [ ] Hyperparameter optimization (Optuna)
- [ ] SHAP explainability
- [ ] GitHub Actions automation
- [ ] Streamlit dashboard
- [ ] Docker containerization
- [ ] AWS EC2 deployment

### 🔮 Phase 3: Advanced (Optional)
- [ ] Ensemble with LSTM
- [ ] Regime detection (HMM)
- [ ] Alternative targets (volatility, sector rotation)
- [ ] Real-time inference API

---

## 🧪 Testing

```bash
pytest tests/
```

**Test Coverage:**
- Data pipeline edge cases (missing data, API failures)
- Feature engineering correctness
- Time-series split validation
- Model inference logic

---

## 📚 Key Learnings

### What Makes This Project Stand Out

1. **No pre-cleaned datasets** - Demonstrates real-world data wrangling
2. **Domain expertise** - Bridges finance knowledge with ML engineering
3. **Production-ready** - Not just a notebook, but a deployed system
4. **Explainability** - SHAP values show *why* predictions were made
5. **Proper validation** - Walk-forward splits, not random shuffling

### Skills Demonstrated

- ✅ API integration (yfinance, FRED)
- ✅ Time-series modeling (stationarity, validation)
- ✅ Feature engineering (technical indicators, macro lags)
- ✅ MLOps (MLflow, automation, containerization)
- ✅ Cloud deployment (AWS EC2)
- ✅ Software engineering (clean code, testing, logging)

---

## 🤝 Contributing

This is a personal learning project, but suggestions are welcome! Open an issue or reach out:

- **Email:** samvieiragarcia@gmail.com
- **GitHub:** [@samf0rd](https://github.com/samf0rd)
- **LinkedIn:** [Samuel Garcia](https://www.linkedin.com/in/samuel-garcia)

---

## 📄 License

MIT License - Feel free to use this as inspiration for your own projects!

---

## 🙏 Acknowledgments

- **Data Sources:** Yahoo Finance, Federal Reserve Economic Data (FRED)
- **Inspiration:** Quantitative finance research, Kaggle competitions
- **Mentorship:** Claude AI (iterative learning partner)

---

**Built with ❤️ and ☕ in Lisbon, Portugal**
