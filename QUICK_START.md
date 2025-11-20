# Quick Start Guide

## 🚀 Fast Setup (5 Minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy example file
copy .env.example .env

# Edit .env with your credentials
notepad .env
```

Required in `.env`:
- `HISTORICAL_DATA_API_KEY` - Your SmartAPI key
- `LIVE_DATA_API_KEY` - Your live API key  
- `CLIENT_CODE` - Your client code
- `PIN` - Your PIN
- `TOTP_SECRET` - Your TOTP secret
- `TELEGRAM_BOT_TOKEN` - From @BotFather
- `TELEGRAM_CHAT_ID` - Your chat ID

### 3. Prepare & Train Model
```bash
python main.py --full
```

This will:
- Fetch 90 days of historical data
- Generate features
- Train the model
- Save to `models/model.pkl`

### 4. Test Telegram
```bash
python main.py --test-telegram
```

### 5. Run Live Engine
```bash
python main.py --live
```

## 📋 Step-by-Step Commands

```bash
# Step 1: Prepare training data only
python main.py --prepare-data

# Step 2: Train model
python main.py --train

# Step 3: Test Telegram
python main.py --test-telegram

# Step 4: Run live predictions
python main.py --live
```

## 🎯 What to Expect

### Training Output
```
PREPARING TRAINING DATA
Processing BANKNIFTY...
Fetched 15000 candles
Cleaned data: 14950 candles
Generated features: 14900 candles
Final data: 14850 candles with target

TRAINING ML MODEL
Training Random Forest model...
Accuracy: 0.6234
Model saved to models/model.pkl
```

### Live Engine Output
```
LIVE TRADING ENGINE STARTED
Prediction interval: 60 seconds
Symbols: BANKNIFTY, NIFTY50

PREDICTION CYCLE - 2024-01-15 14:30:00
Processing BANKNIFTY...
Prediction: BUY | Confidence: 75.00% | Price: 45000.50
Signal sent for BANKNIFTY: BUY (75.00%)
```

### Telegram Message
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

## ⚠️ Common Issues

1. **"Missing credentials"** → Check `.env` file
2. **"Model not found"** → Run `python main.py --train` first
3. **"Telegram error"** → Test with `python main.py --test-telegram`
4. **"No data"** → Check market hours (9:15 AM - 3:30 PM IST)

## 📁 Important Files

- `config.py` - All configuration
- `data/training_ready_data.csv` - Training dataset
- `models/model.pkl` - Trained model
- `logs/trading_signal_system.log` - System logs

## 🔄 Daily Workflow

1. Start engine: `python main.py --live`
2. Monitor Telegram for signals
3. Check logs if issues: `logs/trading_signal_system.log`

## 📚 Full Documentation

See `README_AI_TRADING_SYSTEM.md` for complete documentation.

