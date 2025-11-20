"""
Target Generator Module
Generates target variable for ML model training
"""

import pandas as pd
import numpy as np
from logzero import logger


class TargetGenerator:
    """Generates target variable for classification/regression"""
    
    @staticmethod
    def generate_binary_target(df, lookahead=1, threshold=0.0):
        """
        Generate binary target: 1 for BUY (price goes up), 0 for SELL (price goes down)
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Number of candles to look ahead (default: 1 for next candle)
            threshold: Minimum price change threshold (default: 0.0 for any change)
        
        Returns:
            Series with target values (1 = BUY, 0 = SELL)
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to target generator")
            return pd.Series()
        
        # Calculate future price change
        future_close = df['close'].shift(-lookahead)
        current_close = df['close']
        
        # Calculate percentage change
        price_change = ((future_close - current_close) / current_close) * 100
        
        # Generate binary target
        # 1 = BUY (price goes up), 0 = SELL (price goes down or stays same)
        target = (price_change > threshold).astype(int)
        
        # Remove last 'lookahead' rows (no future data available)
        target.iloc[-lookahead:] = np.nan
        
        logger.info(f"Generated binary target: {target.sum()} BUY signals, {len(target) - target.sum() - target.isna().sum()} SELL signals")
        
        return target
    
    @staticmethod
    def generate_multi_class_target(df, lookahead=1, thresholds=[-0.5, 0.5]):
        """
        Generate multi-class target: 0 = STRONG_SELL, 1 = SELL, 2 = HOLD, 3 = BUY, 4 = STRONG_BUY
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Number of candles to look ahead
            thresholds: List of thresholds for classification (default: [-0.5, 0.5])
        
        Returns:
            Series with target values (0-4)
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to target generator")
            return pd.Series()
        
        # Calculate future price change
        future_close = df['close'].shift(-lookahead)
        current_close = df['close']
        price_change = ((future_close - current_close) / current_close) * 100
        
        # Generate multi-class target
        target = pd.Series(index=df.index, dtype=int)
        target[price_change < thresholds[0]] = 0  # STRONG_SELL
        target[(price_change >= thresholds[0]) & (price_change < 0)] = 1  # SELL
        target[(price_change >= 0) & (price_change < thresholds[1])] = 2  # HOLD
        target[(price_change >= thresholds[1]) & (price_change < thresholds[1] * 2)] = 3  # BUY
        target[price_change >= thresholds[1] * 2] = 4  # STRONG_BUY
        
        # Remove last 'lookahead' rows
        target.iloc[-lookahead:] = np.nan
        
        logger.info(f"Generated multi-class target")
        logger.info(f"  STRONG_SELL (0): {(target == 0).sum()}")
        logger.info(f"  SELL (1): {(target == 1).sum()}")
        logger.info(f"  HOLD (2): {(target == 2).sum()}")
        logger.info(f"  BUY (3): {(target == 3).sum()}")
        logger.info(f"  STRONG_BUY (4): {(target == 4).sum()}")
        
        return target
    
    @staticmethod
    def generate_regression_target(df, lookahead=1):
        """
        Generate regression target: future price change percentage
        
        Args:
            df: DataFrame with OHLCV data
            lookahead: Number of candles to look ahead
        
        Returns:
            Series with price change percentages
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided to target generator")
            return pd.Series()
        
        # Calculate future price change percentage
        future_close = df['close'].shift(-lookahead)
        current_close = df['close']
        price_change_pct = ((future_close - current_close) / current_close) * 100
        
        # Remove last 'lookahead' rows
        price_change_pct.iloc[-lookahead:] = np.nan
        
        logger.info(f"Generated regression target: mean={price_change_pct.mean():.4f}%, std={price_change_pct.std():.4f}%")
        
        return price_change_pct


if __name__ == "__main__":
    # Test target generator
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.data_fetcher import DataFetcher
    from data_pipeline.data_cleaner import DataCleaner
    from feature_engineering.feature_generator import FeatureGenerator
    
    fetcher = DataFetcher()
    if fetcher.connect():
        df = fetcher.fetch_training_data('BANKNIFTY', days=5)
        if df is not None:
            cleaner = DataCleaner()
            cleaned_df = cleaner.clean_ohlcv(df)
            
            generator = FeatureGenerator()
            featured_df = generator.generate_features(cleaned_df)
            
            target_gen = TargetGenerator()
            target = target_gen.generate_binary_target(featured_df)
            
            featured_df['target'] = target
            print(f"\nTarget distribution:")
            print(featured_df['target'].value_counts())
            print(f"\nDataFrame with target:")
            print(featured_df[['timestamp', 'close', 'target']].tail(20))

