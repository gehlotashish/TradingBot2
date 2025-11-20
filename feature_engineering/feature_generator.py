"""
Feature Engineering Module
Generates technical indicators and price-based features
"""

import pandas as pd
import numpy as np
from logzero import logger
import sys
import os

# Try to import ta-lib, fallback to ta library
try:
    import talib
    USE_TALIB = True
except ImportError:
    try:
        import ta
        USE_TALIB = False
    except ImportError:
        logger.warning("Neither talib nor ta library found. Some indicators may not work.")
        USE_TALIB = None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_CONFIG


class FeatureGenerator:
    """Generates technical indicators and engineered features"""
    
    def __init__(self):
        self.ema_short = FEATURE_CONFIG['EMA_SHORT']
        self.ema_long = FEATURE_CONFIG['EMA_LONG']
        self.rsi_period = FEATURE_CONFIG['RSI_PERIOD']
        self.macd_fast = FEATURE_CONFIG['MACD_FAST']
        self.macd_slow = FEATURE_CONFIG['MACD_SLOW']
        self.macd_signal = FEATURE_CONFIG['MACD_SIGNAL']
        self.atr_period = FEATURE_CONFIG['ATR_PERIOD']
        self.volatility_window = FEATURE_CONFIG['VOLATILITY_WINDOW']
        self.volume_ma_window = FEATURE_CONFIG['VOLUME_MA_WINDOW']
    
    def calculate_ema(self, df, period, column='close'):
        """Calculate Exponential Moving Average"""
        return df[column].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, df, period=None):
        """Calculate Relative Strength Index"""
        if period is None:
            period = self.rsi_period
        
        if USE_TALIB:
            return pd.Series(talib.RSI(df['close'].values, timeperiod=period), index=df.index)
        else:
            # Manual RSI calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_macd(self, df):
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if USE_TALIB:
            macd, signal, hist = talib.MACD(
                df['close'].values,
                fastperiod=self.macd_fast,
                slowperiod=self.macd_slow,
                signalperiod=self.macd_signal
            )
            return (
                pd.Series(macd, index=df.index),
                pd.Series(signal, index=df.index),
                pd.Series(hist, index=df.index)
            )
        else:
            # Manual MACD calculation
            ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
            hist = macd - signal
            return macd, signal, hist
    
    def calculate_atr(self, df, period=None):
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period
        
        if USE_TALIB:
            return pd.Series(talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period), index=df.index)
        else:
            # Manual ATR calculation
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window=period).mean()
            return atr
    
    def calculate_volatility(self, df, window=None):
        """Calculate price volatility (standard deviation of returns)"""
        if window is None:
            window = self.volatility_window
        
        returns = df['close'].pct_change()
        volatility = returns.rolling(window=window).std()
        return volatility
    
    def calculate_candle_features(self, df):
        """Calculate candle body and wick features"""
        # Candle body (absolute difference between open and close)
        df['candle_body'] = abs(df['close'] - df['open'])
        
        # Upper wick (high - max(open, close))
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Lower wick (min(open, close) - low)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Total wick
        df['total_wick'] = df['upper_wick'] + df['lower_wick']
        
        # Wick ratio (wick / body, avoid division by zero)
        df['wick_ratio'] = np.where(
            df['candle_body'] > 0,
            df['total_wick'] / df['candle_body'],
            0
        )
        
        return df
    
    def calculate_price_features(self, df):
        """Calculate price-based features"""
        # Price change
        df['price_change'] = df['close'] - df['open']
        
        # Percentage change
        df['change_pct'] = df['close'].pct_change() * 100
        
        # High-Low ratio
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Close position in range (0-1)
        range_size = df['high'] - df['low']
        df['close_position'] = np.where(
            range_size > 0,
            (df['close'] - df['low']) / range_size,
            0.5
        )
        
        return df
    
    def calculate_volume_features(self, df):
        """Calculate volume-based features"""
        # Volume moving average
        df['volume_ma'] = df['volume'].rolling(window=self.volume_ma_window).mean()
        
        # Volume ratio (current volume / average volume)
        df['volume_ratio'] = np.where(
            df['volume_ma'] > 0,
            df['volume'] / df['volume_ma'],
            1.0
        )
        
        return df
    
    def generate_features(self, df):
        """
        Generate all features for the DataFrame
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        
        Returns:
            DataFrame with all features added
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to feature generator")
            return pd.DataFrame()
        
        df = df.copy()
        original_len = len(df)
        
        logger.info(f"Generating features for {len(df)} candles")
        
        # Calculate technical indicators
        df['ema_9'] = self.calculate_ema(df, self.ema_short)
        df['ema_21'] = self.calculate_ema(df, self.ema_long)
        df['rsi'] = self.calculate_rsi(df)
        
        macd, macd_signal, macd_hist = self.calculate_macd(df)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        df['atr'] = self.calculate_atr(df)
        df['volatility'] = self.calculate_volatility(df)
        
        # Calculate candle features
        df = self.calculate_candle_features(df)
        
        # Calculate price features
        df = self.calculate_price_features(df)
        
        # Calculate volume features
        df = self.calculate_volume_features(df)
        
        # Remove rows with NaN (from indicator calculations)
        df = df.dropna().reset_index(drop=True)
        
        final_len = len(df)
        if original_len != final_len:
            logger.info(f"Feature generation: {original_len} -> {final_len} rows (removed NaN rows)")
        
        return df
    
    def get_feature_columns(self):
        """Get list of feature column names"""
        return [
            'open', 'high', 'low', 'close', 'volume',
            'ema_9', 'ema_21', 'rsi', 'macd', 'macd_signal', 'macd_hist',
            'atr', 'volatility', 'candle_body', 'wick_ratio', 'change_pct',
            'volume_ma', 'volume_ratio', 'price_change', 'high_low_ratio'
        ]


if __name__ == "__main__":
    # Test feature generator
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.data_fetcher import DataFetcher
    from data_pipeline.data_cleaner import DataCleaner
    
    fetcher = DataFetcher()
    if fetcher.connect():
        df = fetcher.fetch_training_data('BANKNIFTY', days=5)
        if df is not None:
            cleaner = DataCleaner()
            cleaned_df = cleaner.clean_ohlcv(df)
            
            generator = FeatureGenerator()
            featured_df = generator.generate_features(cleaned_df)
            
            print(f"\nGenerated features for {len(featured_df)} candles")
            print(f"\nFeature columns: {generator.get_feature_columns()}")
            print(f"\nDataFrame shape: {featured_df.shape}")
            print(f"\nFirst few rows:")
            print(featured_df[['timestamp', 'close', 'ema_9', 'ema_21', 'rsi', 'macd', 'atr']].head())
            print(f"\nLast few rows:")
            print(featured_df[['timestamp', 'close', 'ema_9', 'ema_21', 'rsi', 'macd', 'atr']].tail())

