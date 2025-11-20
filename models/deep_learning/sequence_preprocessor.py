"""
Sequence Preprocessor for Deep Learning Models
Converts OHLCV data into sequences for LSTM/GRU training
"""

import pandas as pd
import numpy as np
from logzero import logger
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG


class SequencePreprocessor:
    """Prepares sequence data for deep learning models"""
    
    def __init__(self, sequence_length=60, scaler_type='minmax'):
        """
        Initialize sequence preprocessor
        
        Args:
            sequence_length: Number of candles in each sequence (default: 60)
            scaler_type: 'minmax' or 'standard' (default: 'minmax')
        """
        self.sequence_length = sequence_length
        self.scaler_type = scaler_type
        self.scaler = MinMaxScaler() if scaler_type == 'minmax' else StandardScaler()
        self.feature_columns = MODEL_CONFIG['FEATURE_COLUMNS']
        self.target_column = MODEL_CONFIG['TARGET_COLUMN']
        self.is_fitted = False
    
    def create_sequences(self, df):
        """
        Create sequences from DataFrame
        
        Args:
            df: DataFrame with features and target, sorted by timestamp
        
        Returns:
            X: Array of sequences (samples, sequence_length, features)
            y: Array of targets (samples,)
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided")
            return None, None
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select features
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing feature columns: {missing_cols}")
            return None, None
        
        # Get feature data
        feature_data = df[self.feature_columns].values
        target_data = df[self.target_column].values
        
        # Remove NaN rows
        valid_mask = ~(np.isnan(feature_data).any(axis=1) | np.isnan(target_data))
        feature_data = feature_data[valid_mask]
        target_data = target_data[valid_mask]
        
        if len(feature_data) < self.sequence_length:
            logger.error(f"Insufficient data: {len(feature_data)} < {self.sequence_length}")
            return None, None
        
        # Scale features
        if not self.is_fitted:
            feature_data = self.scaler.fit_transform(feature_data)
            self.is_fitted = True
        else:
            feature_data = self.scaler.transform(feature_data)
        
        # Verify no NaN/Inf after scaling
        if np.isnan(feature_data).any() or np.isinf(feature_data).any():
            logger.warning("NaN/Inf values after scaling, replacing with 0")
            feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - self.sequence_length):
            # Sequence of features
            X.append(feature_data[i:i + self.sequence_length])
            # Target (next candle direction)
            y.append(target_data[i + self.sequence_length])
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Created {len(X)} sequences of length {self.sequence_length}")
        logger.info(f"Sequence shape: {X.shape}")
        logger.info(f"Target distribution: BUY={np.sum(y==1)}, SELL={np.sum(y==0)}")
        
        return X, y
    
    def prepare_for_prediction(self, df):
        """
        Prepare last sequence for prediction
        
        Args:
            df: DataFrame with features, sorted by timestamp
        
        Returns:
            X: Single sequence array (1, sequence_length, features)
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided")
            return None
        
        # Ensure sorted by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Select features
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing feature columns: {missing_cols}")
            return None
        
        # Get last sequence_length rows
        if len(df) < self.sequence_length:
            logger.warning(f"Not enough data: {len(df)} < {self.sequence_length}")
            return None
        
        feature_data = df[self.feature_columns].tail(self.sequence_length).values
        
        # Remove NaN
        if np.isnan(feature_data).any():
            logger.warning("NaN values in feature data, filling with forward fill")
            feature_data = pd.DataFrame(feature_data).fillna(method='ffill').fillna(method='bfill').values
        
        # Scale features
        if not self.is_fitted:
            logger.warning("Scaler not fitted, fitting now")
            self.scaler.fit(feature_data)
            self.is_fitted = True
        
        feature_data = self.scaler.transform(feature_data)
        
        # Reshape to sequence format (1, sequence_length, features)
        X = feature_data.reshape(1, self.sequence_length, len(self.feature_columns))
        
        return X
    
    def fit_scaler(self, df):
        """
        Fit scaler on training data
        
        Args:
            df: Training DataFrame
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame provided")
            return
        
        feature_data = df[self.feature_columns].values
        valid_mask = ~np.isnan(feature_data).any(axis=1)
        feature_data = feature_data[valid_mask]
        
        self.scaler.fit(feature_data)
        self.is_fitted = True
        logger.info("Scaler fitted on training data")
    
    def get_sequence_info(self):
        """Get sequence configuration info"""
        return {
            'sequence_length': self.sequence_length,
            'num_features': len(self.feature_columns),
            'scaler_type': self.scaler_type,
            'is_fitted': self.is_fitted
        }


if __name__ == "__main__":
    # Test sequence preprocessor
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from data_pipeline.data_fetcher import DataFetcher
    from data_pipeline.data_cleaner import DataCleaner
    from feature_engineering.feature_generator import FeatureGenerator
    from models.target_generator import TargetGenerator
    
    # Load training data
    from config import DATA_CONFIG
    import pandas as pd
    
    df = pd.read_csv(DATA_CONFIG['TRAINING_DATA_FILE'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.head(1000)  # Test with subset
    
    # Create sequences
    preprocessor = SequencePreprocessor(sequence_length=60)
    X, y = preprocessor.create_sequences(df)
    
    if X is not None:
        print(f"\nSequences created:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  BUY signals: {np.sum(y==1)}")
        print(f"  SELL signals: {np.sum(y==0)}")

