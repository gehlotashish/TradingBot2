"""
Configuration file for AI Trading Signal System
Contains all settings, API keys, and parameters
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== API CONFIGURATION ====================
# SmartAPI Configuration
SMARTAPI_CONFIG = {
    'API_KEY': os.getenv('HISTORICAL_DATA_API_KEY', os.getenv('API_KEY', '')),
    'LIVE_API_KEY': os.getenv('LIVE_DATA_API_KEY', ''),
    'CLIENT_CODE': os.getenv('CLIENT_CODE', ''),
    'PIN': os.getenv('PIN', ''),
    'TOTP_SECRET': os.getenv('TOTP_SECRET', ''),
}

# Telegram Bot Configuration
TELEGRAM_CONFIG = {
    'BOT_TOKEN': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'CHAT_ID': os.getenv('TELEGRAM_CHAT_ID', ''),
}

# ==================== TRADING SYMBOLS ====================
TRADING_SYMBOLS = {
    'BANKNIFTY': {
        'exchange': 'NSE',
        'token': '99926009',
        'name': 'BANKNIFTY',
        'lot_size': 15
    },
    'NIFTY50': {
        'exchange': 'NSE',
        'token': '99926000',
        'name': 'NIFTY 50',
        'lot_size': 50
    }
}

# ==================== DATA CONFIGURATION ====================
DATA_CONFIG = {
    'INTERVAL': 'ONE_MINUTE',  # ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, etc.
    'HISTORICAL_DAYS': 90,  # Days of historical data for training
    'LIVE_CANDLES': 50,  # Number of candles to keep in memory for live prediction
    'DATA_DIR': 'data',
    'RAW_DATA_DIR': 'data/raw_data',
    'PROCESSED_DATA_DIR': 'data/processed_data',
    'TRAINING_DATA_FILE': 'data/training_ready_data.csv',
}

# ==================== FEATURE ENGINEERING ====================
FEATURE_CONFIG = {
    # Technical Indicators
    'EMA_SHORT': 9,
    'EMA_LONG': 21,
    'RSI_PERIOD': 14,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'ATR_PERIOD': 14,
    
    # Feature Engineering
    'VOLATILITY_WINDOW': 20,
    'VOLUME_MA_WINDOW': 20,
}

# ==================== ML MODEL CONFIGURATION ====================
MODEL_CONFIG = {
    'MODEL_DIR': 'models',
    'MODEL_FILE': 'models/model.pkl',
    'FEATURE_COLUMNS': [
        'open', 'high', 'low', 'close', 'volume',
        'ema_9', 'ema_21', 'rsi', 'macd', 'macd_signal', 'macd_hist',
        'atr', 'volatility', 'candle_body', 'wick_ratio', 'change_pct',
        'volume_ma', 'volume_ratio', 'price_change', 'high_low_ratio'
    ],
    'TARGET_COLUMN': 'target',
    'TEST_SIZE': 0.2,
    'RANDOM_STATE': 42,
}

# ==================== MODEL PARAMETERS ====================
# Random Forest Parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'n_jobs': -1
}

# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

# ==================== SIGNAL GENERATION ====================
SIGNAL_CONFIG = {
    'MIN_CONFIDENCE': 0.65,  # Minimum confidence threshold for signals
    'BREAKOUT_MULTIPLIER': 0.5,  # ATR multiplier for breakout levels
    'BREAKDOWN_MULTIPLIER': 0.5,  # ATR multiplier for breakdown levels
    'SIGNAL_COOLDOWN': 5,  # Minutes to wait before sending another signal
}

# ==================== SCHEDULER CONFIGURATION ====================
SCHEDULER_CONFIG = {
    'PREDICTION_INTERVAL': 60,  # Seconds (1 minute)
    'TRAINING_SCHEDULE': 'daily',  # 'daily', 'weekly', 'manual'
    'TRAINING_TIME': '18:00',  # Time to retrain model (24-hour format)
}

# ==================== LOGGING CONFIGURATION ====================
LOGGING_CONFIG = {
    'LOG_DIR': 'logs',
    'LOG_LEVEL': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'LOG_FILE': 'logs/trading_signal_system.log',
    'MAX_LOG_SIZE': 10 * 1024 * 1024,  # 10MB
    'BACKUP_COUNT': 5,
}

# ==================== VALIDATION ====================
def validate_config():
    """Validate that all required configuration is present"""
    errors = []
    
    # Check SmartAPI config
    if not SMARTAPI_CONFIG['API_KEY']:
        errors.append("SMARTAPI API_KEY is missing")
    if not SMARTAPI_CONFIG['CLIENT_CODE']:
        errors.append("SMARTAPI CLIENT_CODE is missing")
    if not SMARTAPI_CONFIG['PIN']:
        errors.append("SMARTAPI PIN is missing")
    if not SMARTAPI_CONFIG['TOTP_SECRET']:
        errors.append("SMARTAPI TOTP_SECRET is missing")
    
    # Check Telegram config
    if not TELEGRAM_CONFIG['BOT_TOKEN']:
        errors.append("TELEGRAM_BOT_TOKEN is missing")
    if not TELEGRAM_CONFIG['CHAT_ID']:
        errors.append("TELEGRAM_CHAT_ID is missing")
    
    if errors:
        raise ValueError("Configuration errors:\n" + "\n".join(f"- {e}" for e in errors))
    
    return True

