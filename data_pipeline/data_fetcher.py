"""
Data Fetcher Module
Fetches historical and live OHLCV data from SmartAPI
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
from SmartApi import SmartConnect
import pyotp
from logzero import logger
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SMARTAPI_CONFIG, DATA_CONFIG, TRADING_SYMBOLS


class DataFetcher:
    """Fetches OHLCV data from SmartAPI"""
    
    def __init__(self, use_live_api=False):
        """
        Initialize DataFetcher
        
        Args:
            use_live_api: If True, use LIVE_DATA_API_KEY, else use HISTORICAL_DATA_API_KEY
        """
        self.use_live_api = use_live_api
        self.api_key = SMARTAPI_CONFIG['LIVE_API_KEY'] if use_live_api else SMARTAPI_CONFIG['API_KEY']
        self.client_code = SMARTAPI_CONFIG['CLIENT_CODE']
        self.pin = SMARTAPI_CONFIG['PIN']
        self.totp_secret = SMARTAPI_CONFIG['TOTP_SECRET']
        
        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        self.connected = False
        
    def connect(self):
        """Connect to SmartAPI"""
        try:
            if not all([self.api_key, self.client_code, self.pin, self.totp_secret]):
                raise ValueError("Missing SmartAPI credentials in config")
            
            self.smart_api = SmartConnect(self.api_key)
            totp = pyotp.TOTP(self.totp_secret).now()
            
            data = self.smart_api.generateSession(self.client_code, self.pin, totp)
            
            if data['status'] == False:
                raise ConnectionError(f"Authentication failed: {data}")
            
            self.auth_token = data['data']['jwtToken']
            self.feed_token = self.smart_api.getfeedToken()
            self.connected = True
            
            logger.info("Successfully connected to SmartAPI")
            return True
            
        except Exception as e:
            logger.exception(f"Error connecting to SmartAPI: {e}")
            self.connected = False
            return False
    
    def fetch_historical_data(self, symbol_key, interval, from_date, to_date):
        """
        Fetch historical OHLCV data
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS (e.g., 'BANKNIFTY', 'NIFTY50')
            interval: Time interval (e.g., 'ONE_MINUTE', 'FIVE_MINUTE')
            from_date: Start date (datetime or string 'YYYY-MM-DD HH:MM')
            to_date: End date (datetime or string 'YYYY-MM-DD HH:MM')
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            symbol_info = TRADING_SYMBOLS[symbol_key]
            exchange = symbol_info['exchange']
            token = symbol_info['token']
            
            # Format dates
            if isinstance(from_date, datetime):
                from_date_str = from_date.strftime("%Y-%m-%d %H:%M")
            else:
                from_date_str = from_date
            
            if isinstance(to_date, datetime):
                to_date_str = to_date.strftime("%Y-%m-%d %H:%M")
            else:
                to_date_str = to_date
            
            # Prepare request
            historic_param = {
                "exchange": exchange,
                "symboltoken": token,
                "interval": interval,
                "fromdate": from_date_str,
                "todate": to_date_str
            }
            
            logger.info(f"Fetching historical data for {symbol_key} from {from_date_str} to {to_date_str}")
            
            # Fetch data
            response = self.smart_api.getCandleData(historic_param)
            
            if response['status'] == False:
                logger.error(f"API Error: {response}")
                return None
            
            data = response.get('data', [])
            
            if not data:
                logger.warning(f"No data returned for {symbol_key}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Convert to numeric
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any rows with NaN
            df = df.dropna()
            
            logger.info(f"Fetched {len(df)} candles for {symbol_key}")
            return df
            
        except Exception as e:
            logger.exception(f"Error fetching historical data: {e}")
            return None
    
    def fetch_training_data(self, symbol_key, days=None):
        """
        Fetch historical data for training
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS
            days: Number of days of historical data (default from config)
        
        Returns:
            DataFrame with OHLCV data
        """
        if days is None:
            days = DATA_CONFIG['HISTORICAL_DAYS']
        
        to_date = datetime.now()
        from_date = to_date - timedelta(days=days)
        
        # Adjust to market hours (9:15 AM to 3:30 PM IST)
        from_date = from_date.replace(hour=9, minute=15, second=0, microsecond=0)
        to_date = to_date.replace(hour=15, minute=30, second=0, microsecond=0)
        
        return self.fetch_historical_data(
            symbol_key=symbol_key,
            interval=DATA_CONFIG['INTERVAL'],
            from_date=from_date,
            to_date=to_date
        )
    
    def fetch_live_candles(self, symbol_key, num_candles=50):
        """
        Fetch recent candles for live prediction
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS
            num_candles: Number of recent candles to fetch
        
        Returns:
            DataFrame with recent OHLCV data
        """
        # Calculate time range for num_candles (assuming 1-minute candles)
        to_date = datetime.now()
        from_date = to_date - timedelta(minutes=num_candles + 10)  # Add buffer
        
        return self.fetch_historical_data(
            symbol_key=symbol_key,
            interval=DATA_CONFIG['INTERVAL'],
            from_date=from_date,
            to_date=to_date
        )
    
    def disconnect(self):
        """Disconnect from SmartAPI"""
        self.connected = False
        self.smart_api = None
        self.auth_token = None
        self.feed_token = None
        logger.info("Disconnected from SmartAPI")


if __name__ == "__main__":
    # Test data fetcher
    fetcher = DataFetcher()
    if fetcher.connect():
        # Test fetch for BANKNIFTY
        df = fetcher.fetch_training_data('BANKNIFTY', days=7)
        if df is not None:
            print(f"\nFetched {len(df)} candles")
            print(df.head())
            print(df.tail())
        fetcher.disconnect()

