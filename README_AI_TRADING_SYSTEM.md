# AI Trading Signal System

A complete AI-powered trading signal generator system for BankNifty and Nifty50 that uses machine learning to predict BUY/SELL signals and sends alerts via Telegram.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Data Flow](#data-flow)
- [ML Pipeline](#ml-pipeline)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system:
- Fetches live and historical OHLCV data from Angel One SmartAPI
- Generates technical indicators and engineered features
- Trains ML models (RandomForest/XGBoost) on historical data
- Makes real-time predictions with confidence scores
- Detects breakout/breakdown levels
- Sends formatted signals to Telegram

## 🏗️ Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    AI Trading Signal System                     │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Data Pipeline│    │ Feature Eng. │    │ ML Training  │
│              │    │              │    │              │
│ - Fetcher    │───▶│ - Indicators │───▶│ - Target Gen │
│ - Cleaner    │    │ - Features   │    │ - Trainer    │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        │                     │                     ▼
        │                     │              ┌──────────────┐
        │                     │              │ Model Storage│
        │                     │              │ (model.pkl)  │
        │                     │              └──────────────┘
        │                     │
        ▼                     ▼
┌──────────────────────────────────────┐
│      Live Prediction Engine           │
│                                       │
│  ┌──────────────┐    ┌──────────────┐│
│  │ Prediction   │───▶│ Signal Format││
│  │ Engine       │    │              ││
│  └──────────────┘    └──────────────┘│
│         │                    │        │
│         └──────────┬─────────┘        │
│                    ▼                  │
│            ┌──────────────┐           │
│            │ Telegram Bot │           │
│            └──────────────┘           │
└───────────────────────────────────────┘
```

### Data Flow Diagram (DFD)

```
┌─────────────┐
│  SmartAPI   │
│  (Historical)│
└──────┬──────┘
       │ OHLCV Data
       ▼
┌─────────────┐
│ Data Fetcher│
└──────┬──────┘
       │ Raw Data
       ▼
┌─────────────┐
│Data Cleaner │
└──────┬──────┘
       │ Cleaned Data
       ▼
┌─────────────┐
│   Feature   │
│  Generator  │
└──────┬──────┘
       │ Features
       ▼
┌─────────────┐
│   Target    │
│  Generator  │
└──────┬──────┘
       │ Training Data
       ▼
┌─────────────┐
│   Model     │
│   Trainer   │
└──────┬──────┘
       │ Trained Model
       ▼
┌─────────────┐
│ model.pkl   │
└─────────────┘

LIVE PREDICTION FLOW:
┌─────────────┐
│  SmartAPI   │
│   (Live)    │
└──────┬──────┘
       │ Recent Candles
       ▼
┌─────────────┐
│ Data Fetcher│
└──────┬──────┘
       │
       ▼
┌─────────────┐    ┌─────────────┐
│   Feature   │───▶│  Prediction │
│  Generator  │    │   Engine    │
└─────────────┘    └──────┬──────┘
                          │ Signal
                          ▼
                   ┌─────────────┐
                   │  Telegram   │
                   │     Bot     │
                   └─────────────┘
```

### ML Pipeline Architecture

```
TRAINING PHASE:
┌─────────────────────────────────────────────────────────┐
│ 1. Data Collection                                       │
│    - Fetch 90 days historical OHLCV                      │
│    - Clean and validate                                  │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Feature Engineering                                  │
│    - EMA (9, 21)                                        │
│    - RSI (14)                                           │
│    - MACD (12, 26, 9)                                   │
│    - ATR (14)                                           │
│    - Volatility                                         │
│    - Candle body, wick ratio                            │
│    - Volume features                                    │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Target Generation                                    │
│    - Binary: 1=BUY (price up), 0=SELL (price down)      │
│    - Based on next candle direction                      │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Model Training                                        │
│    - RandomForest / XGBoost                              │
│    - Train/Test split (80/20)                           │
│    - Feature importance analysis                         │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Model Evaluation & Storage                           │
│    - Accuracy, Classification Report                     │
│    - Save to model.pkl                                  │
└─────────────────────────────────────────────────────────┘

INFERENCE PHASE:
┌─────────────────────────────────────────────────────────┐
│ 1. Fetch Last 50 Candles                                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Generate Same Features                                │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Load Model & Predict                                 │
│    - Get BUY/SELL signal                                 │
│    - Calculate confidence (probability)                  │
│    - Calculate trigger price (ATR-based)                 │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Format & Send Signal                                 │
│    - Check confidence threshold (≥65%)                   │
│    - Check cooldown period                              │
│    - Send to Telegram                                    │
└─────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Multi-Symbol Support**: BankNifty and Nifty50
- **Technical Indicators**: EMA, RSI, MACD, ATR, Volatility
- **Feature Engineering**: Candle patterns, volume analysis, price ratios
- **ML Models**: RandomForest and XGBoost
- **Real-time Predictions**: 1-minute interval predictions
- **Confidence Scoring**: Probability-based signal confidence
- **Breakout Detection**: ATR-based trigger prices
- **Telegram Integration**: Formatted signal alerts
- **Cooldown Mechanism**: Prevents signal spam
- **Comprehensive Logging**: Full system logging

## 📁 Project Structure

```
ai-trading-signal-system/
│
├── config.py                      # Configuration file
├── main.py                         # Main entry point
├── train_model.py                  # Training script
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── README_AI_TRADING_SYSTEM.md     # This file
│
├── data_pipeline/                  # Data collection and cleaning
│   ├── __init__.py
│   ├── data_fetcher.py            # SmartAPI data fetcher
│   └── data_cleaner.py            # Data cleaning and validation
│
├── feature_engineering/            # Feature generation
│   ├── __init__.py
│   └── feature_generator.py        # Technical indicators & features
│
├── models/                         # ML models
│   ├── __init__.py
│   ├── target_generator.py         # Target variable generation
│   └── model_trainer.py            # Model training and evaluation
│
├── live_engine/                    # Live prediction system
│   ├── __init__.py
│   ├── prediction_engine.py        # Real-time predictions
│   └── live_trading_engine.py     # Main live engine
│
├── telegram_bot/                    # Telegram integration
│   ├── __init__.py
│   └── telegram_sender.py         # Telegram message sender
│
├── data/                           # Data storage
│   ├── raw_data/                   # Raw historical data
│   ├── processed_data/             # Processed data
│   └── training_ready_data.csv     # Final training dataset
│
├── models/                         # Model storage
│   └── model.pkl                   # Trained model file
│
└── logs/                           # Log files
    └── trading_signal_system.log
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.8 or higher
- SmartAPI account with API keys
- Telegram bot token and chat ID

### Step 1: Clone/Download Project

```bash
cd "c:\New folder\historical data"
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note for TA-Lib**: 
- Option 1 (Recommended): Install TA-Lib C library first from https://ta-lib.org/install/, then `pip install TA-Lib`
- Option 2 (Easier): Use `ta` library (already in requirements.txt) - pure Python implementation

### Step 4: Configure Environment Variables

1. Copy `.env.example` to `.env`:
```bash
copy .env.example .env
```

2. Edit `.env` and add your credentials:

```env
# SmartAPI Configuration
HISTORICAL_DATA_API_KEY=your_historical_api_key
LIVE_DATA_API_KEY=your_live_api_key
CLIENT_CODE=your_client_code
PIN=your_pin
TOTP_SECRET=your_totp_secret

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Step 5: Get Telegram Bot Token and Chat ID

1. **Create Telegram Bot**:
   - Open Telegram, search for `@BotFather`
   - Send `/newbot` and follow instructions
   - Copy the bot token

2. **Get Chat ID**:
   - Search for `@userinfobot` on Telegram
   - Start a conversation, it will show your chat ID
   - Or create a group, add your bot, and get group chat ID

### Step 6: Prepare Training Data

```bash
python main.py --prepare-data
```

This will:
- Fetch 90 days of historical data for BankNifty and Nifty50
- Clean and validate the data
- Generate all features
- Create target variable
- Save to `data/training_ready_data.csv`

### Step 7: Train Model

```bash
# Train RandomForest (default)
python main.py --train

# Or train XGBoost
python main.py --train --model-type xgboost

# Or do both: prepare data and train
python main.py --full
```

### Step 8: Test Telegram Configuration

```bash
python main.py --test-telegram
```

You should receive a test message on Telegram.

### Step 9: Run Live Trading Engine

```bash
python main.py --live
```

The engine will:
- Load the trained model
- Fetch live data every 1 minute
- Generate predictions
- Send signals to Telegram (if confidence ≥ 65%)

## 📊 Usage

### Command Line Options

```bash
# Prepare training data only
python main.py --prepare-data

# Train model only (requires training data)
python main.py --train

# Prepare data and train
python main.py --full

# Run live engine
python main.py --live

# Test Telegram
python main.py --test-telegram

# Specify model type
python main.py --train --model-type xgboost
```

### Training Data Format

The training dataset (`data/training_ready_data.csv`) contains:

**Columns:**
- `timestamp`: Datetime
- `open`, `high`, `low`, `close`, `volume`: OHLCV data
- `ema_9`, `ema_21`: Exponential Moving Averages
- `rsi`: Relative Strength Index
- `macd`, `macd_signal`, `macd_hist`: MACD indicators
- `atr`: Average True Range
- `volatility`: Price volatility
- `candle_body`, `wick_ratio`: Candle features
- `change_pct`: Price change percentage
- `volume_ma`, `volume_ratio`: Volume features
- `price_change`, `high_low_ratio`: Price features
- `target`: Binary target (1=BUY, 0=SELL)
- `symbol`: Symbol identifier (BANKNIFTY/NIFTY50)

### Signal Format

Telegram messages are sent in this format:

```
🤖 AI MODULE: ACTIVE

📊 Index: BANKNIFTY
💰 Current Price: ₹45000.50
🟢 Signal: BUY
🎯 Trigger: ₹45100.00
📈 Accuracy: 75.0%

💡 Reason: RSI oversold, MACD bullish, EMA bullish crossover

⏰ Time: 2024-01-15 14:30:00
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Trading Symbols**: Add/modify symbols in `TRADING_SYMBOLS`
- **Feature Parameters**: EMA periods, RSI period, MACD parameters, etc.
- **Model Parameters**: RandomForest/XGBoost hyperparameters
- **Signal Thresholds**: Minimum confidence, cooldown period
- **Scheduler**: Prediction interval, training schedule

## 🔍 Troubleshooting

### Common Issues

1. **"Missing credentials" error**
   - Check `.env` file exists and has all required variables
   - Ensure no extra spaces in values

2. **"Failed to connect to SmartAPI"**
   - Verify API keys are correct
   - Check TOTP secret is valid
   - Ensure account has API access enabled

3. **"Model file not found"**
   - Run training first: `python main.py --train`
   - Check `models/model.pkl` exists

4. **"Telegram message not sent"**
   - Verify bot token and chat ID
   - Test with `python main.py --test-telegram`
   - Check bot is started in Telegram

5. **"No data fetched"**
   - Check market hours (9:15 AM - 3:30 PM IST)
   - Verify symbol tokens are correct
   - Check API rate limits

6. **Import errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`
   - For TA-Lib, install C library first

### Logs

Check `logs/trading_signal_system.log` for detailed error messages.

## 📈 Model Performance

After training, you'll see:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Feature importance rankings

Typical performance:
- RandomForest: 55-65% accuracy
- XGBoost: 60-70% accuracy

**Note**: Higher accuracy doesn't always mean better trading performance. Consider:
- Signal quality (confidence scores)
- Risk-reward ratios
- Market conditions
- Backtesting results

## 🔄 Workflow

### Initial Setup (One-time)
1. Install dependencies
2. Configure `.env`
3. Prepare training data
4. Train model
5. Test Telegram

### Daily Operation
1. Start live engine: `python main.py --live`
2. Monitor Telegram for signals
3. Review logs for issues

### Weekly/Monthly
1. Retrain model with fresh data
2. Update feature parameters if needed
3. Review and adjust signal thresholds

## 📝 Notes

- **Market Hours**: System fetches data during market hours (9:15 AM - 3:30 PM IST)
- **Data Storage**: Live data is not stored permanently (only last 50 candles in memory)
- **Rate Limits**: Be aware of SmartAPI rate limits
- **Model Updates**: Retrain model periodically with fresh data
- **Risk Disclaimer**: This is for educational purposes. Always do your own research.

## 🎯 Expected Output

When running live, you'll see:

```
2024-01-15 14:30:00 | INFO | Processing BANKNIFTY...
2024-01-15 14:30:01 | INFO | Prediction: BUY | Confidence: 75.00% | Price: 45000.50
2024-01-15 14:30:02 | INFO | Signal sent for BANKNIFTY: BUY (75.00%)
```

Telegram message format matches the specification:
```
🤖 AI MODULE: ACTIVE
📊 Index: BANKNIFTY
💰 Current Price: ₹45000.50
🟢 Signal: BUY
🎯 Trigger: ₹45100.00
📈 Accuracy: 75.0%
💡 Reason: RSI oversold, MACD bullish
⏰ Time: 2024-01-15 14:30:00
```

## 📚 Module Interaction Diagram

```
main.py
  │
  ├─▶ train_model.py
  │     ├─▶ DataFetcher
  │     ├─▶ DataCleaner
  │     ├─▶ FeatureGenerator
  │     ├─▶ TargetGenerator
  │     └─▶ ModelTrainer
  │
  └─▶ LiveTradingEngine
        ├─▶ PredictionEngine
        │     ├─▶ DataFetcher (live)
        │     ├─▶ DataCleaner
        │     ├─▶ FeatureGenerator
        │     └─▶ ModelTrainer (load model)
        │
        └─▶ TelegramSender
```

## 🔐 Security Notes

- Never commit `.env` file to version control
- Keep API keys secure
- Use environment variables in production
- Rotate API keys periodically

## 📞 Support

For issues or questions:
1. Check logs in `logs/trading_signal_system.log`
2. Review configuration in `config.py`
3. Verify all dependencies are installed
4. Test each component individually

---

**Built with ❤️ for algorithmic trading**

