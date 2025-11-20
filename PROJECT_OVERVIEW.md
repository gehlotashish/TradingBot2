# AI Trading Signal System - Complete Project Overview

## 📋 Project Summary

This is a complete AI-powered trading signal generator system for BankNifty and Nifty50. The system:
- Fetches live and historical OHLCV data from Angel One SmartAPI
- Generates technical indicators and engineered features
- Trains ML models (RandomForest, XGBoost, LSTM, Multi-Head LSTM)
- Predicts BUY/SELL signals with probability scores
- Sends formatted signals to Telegram

## 🏗️ Project Structure

```
historical data/
├── config.py                          # Central configuration
├── main.py                            # Main entry point
├── train_model.py                     # Model training script
├── requirements.txt                   # Dependencies
├── .env                               # Environment variables (API keys)
│
├── data/
│   ├── raw_data/                      # Raw historical data
│   ├── processed_data/                # Processed data
│   └── training_ready_data.csv       # Final training dataset
│
├── data_pipeline/
│   ├── data_fetcher.py               # Fetches data from SmartAPI
│   └── data_cleaner.py               # Cleans OHLCV data
│
├── feature_engineering/
│   └── feature_generator.py          # Generates technical indicators
│
├── models/
│   ├── target_generator.py          # Generates target variables
│   ├── model_trainer.py             # Trains traditional ML models
│   ├── ensemble_model.py            # Combines multiple models
│   │
│   └── deep_learning/
│       ├── sequence_preprocessor.py # Prepares sequences for LSTM
│       ├── lstm_model.py            # Single-head LSTM architecture
│       ├── dl_trainer.py           # LSTM training pipeline
│       ├── multi_head_lstm.py      # Multi-head LSTM (4 outputs)
│       └── multi_head_trainer.py   # Multi-head training pipeline
│
├── live_engine/
│   └── prediction_engine.py         # Real-time prediction engine
│
└── telegram_bot/
    └── telegram_messenger.py       # Sends signals to Telegram
```

## 🎯 Key Features

### 1. Data Pipeline
- **SmartAPI Integration**: Fetches historical and live OHLCV data
- **Data Cleaning**: Handles missing values, outliers, duplicates
- **Multi-Symbol Support**: BankNifty and Nifty50

### 2. Feature Engineering
**Technical Indicators:**
- EMA (9, 21)
- RSI (14)
- MACD (12, 26, 9)
- ATR (14)
- Volatility
- VWAP (Volume Weighted Average Price)

**Engineered Features:**
- Candle body, wick ratio
- Price change, percentage change
- Volume delta, volume momentum
- Trend direction, trend strength
- EMA crossovers

**Total Features: 27**

### 3. Target Generation
- **Binary Classification**: BUY (1) / SELL (0)
- **Multi-Timeframe**: 5-min and 15-min predictions
- **Trend Continuation**: Predicts if trend continues
- **Regression**: Predicts price change magnitude
- **Volatility**: Predicts future volatility

### 4. Machine Learning Models

#### Traditional ML:
- **RandomForest**: 70.6% accuracy
- **XGBoost**: Ensemble model

#### Deep Learning:
- **Single-Head LSTM**: 3 LSTM layers (128→64→32), 2 Dense layers
- **Multi-Head LSTM**: 4 simultaneous outputs:
  - Trend Continuation (classification)
  - Direction (classification)
  - Magnitude (regression)
  - Volatility (regression)

### 5. Live Prediction Engine
- Fetches latest candles
- Generates features
- Loads trained models
- Makes predictions
- Sends Telegram alerts

### 6. Telegram Integration
- Sends formatted trading signals
- Includes confidence scores
- Trigger prices (ATR-based)
- Technical indicator reasons

## 📊 Model Architecture

### Multi-Head LSTM (Hedge Fund Style)
```
Input: (60 candles × 27 features)
    ↓
Shared LSTM Layers:
  - LSTM 128 units (return_sequences=True)
  - BatchNormalization
  - Dropout 0.3
  - LSTM 64 units (return_sequences=True)
  - BatchNormalization
  - Dropout 0.3
  - LSTM 32 units (return_sequences=False)
  - BatchNormalization
  - Dropout 0.3
    ↓
Shared Dense Layer: 64 units
    ↓
    ├─→ Trend Head → Binary (sigmoid)
    ├─→ Direction Head → Binary (sigmoid)
    ├─→ Magnitude Head → Regression (linear)
    └─→ Volatility Head → Regression (linear)
```

## ⚙️ Configuration

### config.py
```python
# Trading Symbols
TRADING_SYMBOLS = {
    'BANKNIFTY': {...},
    'NIFTY50': {...}
}

# Target Configuration
TARGET_CONFIG = {
    'TARGET_TYPE': 'multi_timeframe',  # 15-min prediction
    'LOOKAHEAD_LONG': 15,
    'LOOKAHEAD_SHORT': 5,
}

# Deep Learning
DL_CONFIG = {
    'ENABLED': True,
    'USE_MULTI_HEAD': True,  # Multi-head model
    'SEQUENCE_LENGTH': 60,
}

# LSTM Parameters
LSTM_PARAMS = {
    'lstm_units_1': 128,
    'lstm_units_2': 64,
    'dense_units': 32,
    'dropout_rate': 0.3,
    'learning_rate': 0.0005,
    'batch_size': 64,
    'epochs': 100,
    'use_class_weights': True,
}
```

## 🚀 Usage

### 1. Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
SMARTAPI_API_KEY=your_key
SMARTAPI_CLIENT_ID=your_id
SMARTAPI_PIN=your_pin
```

### 2. Prepare Training Data
```bash
python main.py --prepare-data
```
- Fetches historical data
- Generates features
- Creates targets
- Saves to `data/training_ready_data.csv`

### 3. Train Models
```bash
# Train traditional ML only
python main.py --train

# Train with LSTM
python main.py --train --train-lstm
```

### 4. Run Live Engine
```bash
python main.py --live
```

## 📈 Current Performance

- **RandomForest**: 70.6% accuracy
- **LSTM**: ~50-51% accuracy (learning issue - task difficulty)
- **Multi-Head LSTM**: Architecture ready, needs retraining

## 🔧 Key Files Explained

### config.py
Central configuration for:
- Trading symbols
- Feature parameters
- Model hyperparameters
- API credentials

### train_model.py
Orchestrates:
- Data preparation
- Traditional ML training
- Deep learning training (single/multi-head)
- Model saving

### feature_engineering/feature_generator.py
Generates:
- Technical indicators (EMA, RSI, MACD, ATR)
- Price features (change, volatility)
- Volume features (delta, momentum)
- Trend features (direction, strength)
- VWAP

### models/target_generator.py
Creates targets:
- Binary (BUY/SELL)
- Trend continuation
- Regression (magnitude)
- Volatility

### models/deep_learning/multi_head_lstm.py
Multi-head architecture:
- Shared LSTM layers
- 4 separate output heads
- Different losses for each head
- Class weights for imbalanced data

### live_engine/prediction_engine.py
Real-time prediction:
- Fetches latest candles
- Generates features
- Loads models
- Makes predictions
- Formats signals

## 🐛 Recent Fixes

1. **Feature Generator NaN Bug**: Fixed blind `dropna()` that deleted all data
2. **Multi-Head Indexing**: Fixed index alignment issues
3. **Missing Features**: Auto-fills missing columns in training data
4. **LSTM Learning**: Applied class weights, gradient clipping, better initialization

## 📝 Important Notes

1. **LSTM Accuracy**: Currently ~50% (random guessing)
   - Task difficulty: Predicting next candle is inherently hard
   - 15-min prediction should improve to 55-62%
   - Multi-head model should be more stable

2. **Data Requirements**:
   - Minimum 60 candles for sequence
   - 90 days historical data recommended
   - Features require warm-up period (first 20-30 rows may be NaN)

3. **Model Files**:
   - Traditional ML: `models/model.pkl`
   - Single-head LSTM: `models/lstm_model.h5`
   - Multi-head LSTM: `models/multi_head_lstm_model.h5`

## 🎯 Next Steps

1. Regenerate training data with new features
2. Retrain multi-head model
3. Test live prediction engine
4. Monitor Telegram signals

---

**Project Status**: ✅ Complete Architecture
**Ready for**: Training and Live Trading
**Last Updated**: 2025-11-16

