"""
Data Cleaner Module
Cleans and validates OHLCV data
"""

import pandas as pd
import numpy as np
from logzero import logger


class DataCleaner:
    """Cleans and validates OHLCV data"""
    
    @staticmethod
    def clean_ohlcv(df):
        """
        Clean OHLCV DataFrame
        
        Args:
            df: DataFrame with columns: timestamp, open, high, low, close, volume
        
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            logger.warning("Empty DataFrame provided to cleaner")
            return pd.DataFrame()
        
        df = df.copy()
        original_len = len(df)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate OHLC relationships
        # High should be >= max(open, close)
        # Low should be <= min(open, close)
        invalid_high = df['high'] < df[['open', 'close']].max(axis=1)
        invalid_low = df['low'] > df[['open', 'close']].min(axis=1)
        
        invalid_rows = invalid_high | invalid_low
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} rows with invalid OHLC relationships")
            df = df[~invalid_rows].reset_index(drop=True)
        
        # Remove rows with negative values
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df = df[df[col] >= 0].reset_index(drop=True)
        
        # Remove rows with NaN
        df = df.dropna(subset=numeric_cols).reset_index(drop=True)
        
        # Remove zero volume candles (optional - might want to keep for some analysis)
        # df = df[df['volume'] > 0].reset_index(drop=True)
        
        # Remove outliers (values beyond 3 standard deviations)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df = df[abs(df[col] - mean) <= 3 * std].reset_index(drop=True)
        
        # Fill any remaining NaN values with forward fill
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        cleaned_len = len(df)
        if original_len != cleaned_len:
            logger.info(f"Cleaned data: {original_len} -> {cleaned_len} rows")
        
        return df
    
    @staticmethod
    def validate_data(df):
        """
        Validate cleaned data
        
        Args:
            df: DataFrame to validate
        
        Returns:
            bool: True if valid, False otherwise
        """
        if df is None or df.empty:
            logger.error("DataFrame is empty")
            return False
        
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for NaN values
        if df[required_cols].isna().any().any():
            logger.error("DataFrame contains NaN values")
            return False
        
        # Check OHLC relationships
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1)).any()
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1)).any()
        
        if invalid_high or invalid_low:
            logger.error("Invalid OHLC relationships found")
            return False
        
        logger.info("Data validation passed")
        return True
    
    @staticmethod
    def format_for_training(df):
        """
        Format DataFrame for training
        
        Args:
            df: Cleaned OHLCV DataFrame
        
        Returns:
            Formatted DataFrame ready for feature engineering
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


if __name__ == "__main__":
    # Test data cleaner
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.data_fetcher import DataFetcher
    
    fetcher = DataFetcher()
    if fetcher.connect():
        df = fetcher.fetch_training_data('BANKNIFTY', days=1)
        if df is not None:
            cleaner = DataCleaner()
            cleaned_df = cleaner.clean_ohlcv(df)
            print(f"\nOriginal: {len(df)} rows")
            print(f"Cleaned: {len(cleaned_df)} rows")
            print(cleaned_df.head())
            
            if cleaner.validate_data(cleaned_df):
                print("\n✓ Data validation passed")
            else:
                print("\n✗ Data validation failed")

