# Setup Checklist

Use this checklist to ensure everything is configured correctly before running the system.

## ✅ Pre-Setup

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] Project directory structure created

## ✅ Dependencies

- [ ] `pip install -r requirements.txt` completed successfully
- [ ] TA-Lib or `ta` library installed (check with `python -c "import ta"`)
- [ ] All imports work: `python -c "import pandas, numpy, sklearn, xgboost"`

## ✅ SmartAPI Configuration

- [ ] `.env` file created from `.env.example`
- [ ] `HISTORICAL_DATA_API_KEY` set in `.env`
- [ ] `LIVE_DATA_API_KEY` set in `.env`
- [ ] `CLIENT_CODE` set in `.env`
- [ ] `PIN` set in `.env`
- [ ] `TOTP_SECRET` set in `.env`
- [ ] Tested SmartAPI connection (run a test fetch)

## ✅ Telegram Configuration

- [ ] Telegram bot created via @BotFather
- [ ] `TELEGRAM_BOT_TOKEN` copied to `.env`
- [ ] Chat ID obtained (via @userinfobot or group)
- [ ] `TELEGRAM_CHAT_ID` set in `.env`
- [ ] Tested Telegram: `python main.py --test-telegram`

## ✅ Data Preparation

- [ ] Training data prepared: `python main.py --prepare-data`
- [ ] `data/training_ready_data.csv` file exists
- [ ] Training data has reasonable number of rows (>1000)
- [ ] Training data has all required columns

## ✅ Model Training

- [ ] Model trained: `python main.py --train`
- [ ] `models/model.pkl` file exists
- [ ] Training accuracy logged (check logs)
- [ ] Model evaluation metrics reviewed

## ✅ Live Engine Test

- [ ] Live engine starts without errors: `python main.py --live`
- [ ] Predictions are being generated
- [ ] Telegram messages are being sent (if confidence threshold met)
- [ ] Logs are being written to `logs/trading_signal_system.log`

## ✅ Verification Tests

Run these tests to verify each component:

```bash
# Test 1: Configuration
python -c "from config import validate_config; validate_config(); print('✓ Config OK')"

# Test 2: Data Fetcher
python -c "from data_pipeline.data_fetcher import DataFetcher; f=DataFetcher(); print('✓ DataFetcher OK' if f.connect() else '✗ DataFetcher Failed')"

# Test 3: Feature Generator
python -c "from feature_engineering.feature_generator import FeatureGenerator; print('✓ FeatureGenerator OK')"

# Test 4: Model Load
python -c "from models.model_trainer import ModelTrainer; m=ModelTrainer(); m.load_model(); print('✓ Model Load OK')"

# Test 5: Telegram
python main.py --test-telegram
```

## 🚨 Common Issues to Check

- [ ] No extra spaces in `.env` values
- [ ] Market hours (9:15 AM - 3:30 PM IST) for data fetching
- [ ] API rate limits not exceeded
- [ ] Sufficient disk space for data and models
- [ ] Internet connection stable
- [ ] Firewall not blocking API/Telegram connections

## 📝 Notes

- Keep `.env` file secure (never commit to git)
- Check `logs/trading_signal_system.log` for detailed errors
- Retrain model periodically with fresh data
- Monitor API usage to avoid rate limits

---

Once all items are checked, you're ready to run the live engine!

