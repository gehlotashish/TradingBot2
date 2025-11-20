"""
Multi-Head LSTM Trainer
Handles training pipeline for multi-head models
"""

import os
import numpy as np
import pickle
from logzero import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.utils.class_weight import compute_class_weight
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG, DATA_CONFIG, DL_CONFIG, TARGET_CONFIG
from models.deep_learning.sequence_preprocessor import SequencePreprocessor
from models.deep_learning.multi_head_lstm import MultiHeadLSTMModel
from models.target_generator import TargetGenerator


class MultiHeadDLTrainer:
    """Trains and evaluates multi-head deep learning models"""
    
    def __init__(self, sequence_length=60, model_params=None):
        """
        Initialize Multi-Head DL trainer
        
        Args:
            sequence_length: Length of sequences for LSTM
            model_params: Model hyperparameters
        """
        self.sequence_length = sequence_length
        self.preprocessor = SequencePreprocessor(sequence_length=sequence_length)
        self.multi_head_model = MultiHeadLSTMModel(sequence_length=sequence_length, model_params=model_params)
        self.model_type = 'multi_head_lstm'
        self.is_trained = False
    
    def prepare_multi_targets(self, df):
        """
        Prepare multiple targets for multi-head model
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            Dictionary with all targets (all with same index as df)
        """
        # Reset index first to ensure alignment
        df = df.reset_index(drop=True)
        
        target_gen = TargetGenerator()
        lookahead = TARGET_CONFIG.get('LOOKAHEAD_LONG', 15)
        
        # Generate all targets
        targets = {}
        
        # 1. Trend Continuation
        targets['trend_continuation'] = target_gen.generate_trend_target(
            df, 
            lookahead=lookahead,
            trend_window=TARGET_CONFIG.get('TREND_WINDOW', 5)
        )
        
        # 2. Direction (15-min)
        targets['direction'] = target_gen.generate_binary_target(
            df,
            lookahead=lookahead
        )
        
        # 3. Magnitude (price change)
        targets['magnitude'] = target_gen.generate_regression_target(
            df,
            lookahead=lookahead
        )
        
        # 4. Volatility (future volatility)
        # Calculate future volatility
        future_close = df['close'].shift(-lookahead)
        # Use fill_method=None to avoid deprecation warning
        future_returns = future_close.pct_change(fill_method=None)
        
        # Calculate rolling volatility
        volatility_window = 20
        targets['volatility'] = future_returns.rolling(window=volatility_window).std()
        
        # Ensure all targets have same index as df
        for key in targets:
            targets[key] = targets[key].reset_index(drop=True)
        
        logger.info("Generated multi-head targets:")
        logger.info(f"  Trend continuation: {targets['trend_continuation'].sum()} continuations")
        logger.info(f"  Direction: {targets['direction'].sum()} BUY signals")
        logger.info(f"  Magnitude: mean={targets['magnitude'].mean():.2f}, std={targets['magnitude'].std():.2f}")
        logger.info(f"  Volatility: mean={targets['volatility'].mean():.4f}, std={targets['volatility'].std():.4f}")
        
        return targets
    
    def prepare_data(self, df):
        """
        Prepare sequence data with multiple targets
        
        Args:
            df: DataFrame with features
        
        Returns:
            X_train, X_test, y_train, y_test (dicts)
        """
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided")
        
        logger.info(f"Preparing multi-head data (sequence_length={self.sequence_length})...")
        
        # Generate all targets (already with reset index)
        targets = self.prepare_multi_targets(df)
        
        # Reset df index to match targets
        df = df.reset_index(drop=True)
        
        # Create valid mask (all should have same length now)
        valid_mask = ~(
            targets['trend_continuation'].isna() |
            targets['direction'].isna() |
            targets['magnitude'].isna() |
            targets['volatility'].isna()
        )
        
        # Convert to numpy array for indexing (avoids pandas index issues)
        valid_indices = valid_mask.values if hasattr(valid_mask, 'values') else np.array(valid_mask)
        
        # Filter using numpy boolean indexing
        df_aligned = df.iloc[valid_indices].reset_index(drop=True)
        for key in targets:
            targets[key] = targets[key].iloc[valid_indices].reset_index(drop=True)
        
        # Create sequences
        X, y_dict = [], {
            'trend_continuation': [],
            'direction': [],
            'magnitude': [],
            'volatility': []
        }
        
        # Fit scaler first
        self.preprocessor.fit_scaler(df_aligned)
        
        # Create sequences
        feature_data = df_aligned[MODEL_CONFIG['FEATURE_COLUMNS']].values
        feature_data = self.preprocessor.scaler.transform(feature_data)
        
        for i in range(len(feature_data) - self.sequence_length):
            X.append(feature_data[i:i + self.sequence_length])
            target_idx = i + self.sequence_length
            y_dict['trend_continuation'].append(targets['trend_continuation'].iloc[target_idx])
            y_dict['direction'].append(targets['direction'].iloc[target_idx])
            y_dict['magnitude'].append(targets['magnitude'].iloc[target_idx])
            y_dict['volatility'].append(targets['volatility'].iloc[target_idx])
        
        X = np.array(X)
        for key in y_dict:
            y_dict[key] = np.array(y_dict[key])
        
        logger.info(f"Created {len(X)} sequences")
        logger.debug(f"X shape: {X.shape}")
        for key in y_dict:
            logger.debug(f"y_{key} shape: {y_dict[key].shape}")
        
        # Split into train and test
        # Unpack dictionary values for train_test_split
        y_trend = y_dict['trend_continuation']
        y_direction = y_dict['direction']
        y_magnitude = y_dict['magnitude']
        y_volatility = y_dict['volatility']
        
        X_train, X_test, y_trend_train, y_trend_test, y_dir_train, y_dir_test, y_mag_train, y_mag_test, y_vol_train, y_vol_test = train_test_split(
            X, y_trend, y_direction, y_magnitude, y_volatility,
            test_size=MODEL_CONFIG['TEST_SIZE'],
            random_state=MODEL_CONFIG['RANDOM_STATE']
        )
        
        # Reconstruct dictionaries
        y_train_dict = {
            'trend_continuation': y_trend_train,
            'direction': y_dir_train,
            'magnitude': y_mag_train,
            'volatility': y_vol_train
        }
        
        y_test_dict = {
            'trend_continuation': y_trend_test,
            'direction': y_dir_test,
            'magnitude': y_mag_test,
            'volatility': y_vol_test
        }
        
        logger.info(f"Train sequences: {len(X_train)}")
        logger.info(f"Test sequences: {len(X_test)}")
        
        return X_train, X_test, y_train_dict, y_test_dict
    
    def train(self, df, epochs=100, batch_size=64, validation_split=0.2):
        """
        Train multi-head model
        
        Args:
            df: Training DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        
        Returns:
            Training history and metrics
        """
        logger.info("="*60)
        logger.info("TRAINING MULTI-HEAD LSTM MODEL")
        logger.info("="*60)
        
        # Prepare data
        X_train, X_test, y_train_dict, y_test_dict = self.prepare_data(df)
        
        # Calculate class weights for classification heads
        trend_classes = np.unique(y_train_dict['trend_continuation'])
        direction_classes = np.unique(y_train_dict['direction'])
        
        trend_weights = compute_class_weight('balanced', classes=trend_classes, y=y_train_dict['trend_continuation'])
        direction_weights = compute_class_weight('balanced', classes=direction_classes, y=y_train_dict['direction'])
        
        class_weight = {
            0: (trend_weights[0] + direction_weights[0]) / 2,
            1: (trend_weights[1] + direction_weights[1]) / 2
        }
        
        logger.info(f"Class weights: {class_weight}")
        
        # Update model params
        self.multi_head_model.model_params['epochs'] = epochs
        self.multi_head_model.model_params['batch_size'] = batch_size
        self.multi_head_model.model_params['validation_split'] = validation_split
        
        # Build model
        self.multi_head_model.build_model()
        
        # Train model
        history = self.multi_head_model.train(
            X_train, y_train_dict,
            X_val=None, y_val_dict=None,
            class_weight=class_weight,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self.multi_head_model.evaluate(X_test, y_test_dict)
        
        # Additional evaluation
        predictions = self.multi_head_model.predict(X_test)
        
        # Classification metrics
        trend_pred = (predictions['trend_continuation'] > 0.5).astype(int)
        direction_pred = (predictions['direction'] > 0.5).astype(int)
        
        trend_acc = accuracy_score(y_test_dict['trend_continuation'], trend_pred)
        direction_acc = accuracy_score(y_test_dict['direction'], direction_pred)
        
        # Regression metrics
        magnitude_mae = mean_absolute_error(y_test_dict['magnitude'], predictions['magnitude'])
        volatility_mae = mean_absolute_error(y_test_dict['volatility'], predictions['volatility'])
        
        logger.info(f"\nTest Metrics:")
        logger.info(f"  Trend Continuation Accuracy: {trend_acc:.4f}")
        logger.info(f"  Direction Accuracy: {direction_acc:.4f}")
        logger.info(f"  Magnitude MAE: {magnitude_mae:.4f}")
        logger.info(f"  Volatility MAE: {volatility_mae:.4f}")
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'test_metrics': metrics,
            'trend_accuracy': trend_acc,
            'direction_accuracy': direction_acc,
            'magnitude_mae': magnitude_mae,
            'volatility_mae': volatility_mae,
            'predictions': predictions
        }
    
    def save_model(self, model_dir=None):
        """Save trained model and preprocessor"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if model_dir is None:
            model_dir = MODEL_CONFIG['MODEL_DIR']
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model (use .keras format for better compatibility with custom losses)
        model_path = os.path.join(model_dir, 'multi_head_lstm_model.keras')
        self.multi_head_model.save_model(model_path)
        
        # Save preprocessor
        preprocessor_path = os.path.join(model_dir, 'multi_head_preprocessor.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'scaler': self.preprocessor.scaler,
                'sequence_length': self.preprocessor.sequence_length,
                'scaler_type': self.preprocessor.scaler_type,
                'feature_columns': self.preprocessor.feature_columns,
                'is_fitted': self.preprocessor.is_fitted
            }, f)
        
        logger.info(f"Multi-head model saved to {model_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_dir=None):
        """Load trained model and preprocessor"""
        if model_dir is None:
            model_dir = MODEL_CONFIG['MODEL_DIR']
        
        # Load model (try .keras first, fallback to .h5 for backward compatibility)
        model_path_keras = os.path.join(model_dir, 'multi_head_lstm_model.keras')
        model_path_h5 = os.path.join(model_dir, 'multi_head_lstm_model.h5')
        
        if os.path.exists(model_path_keras):
            model_path = model_path_keras
        elif os.path.exists(model_path_h5):
            model_path = model_path_h5
            logger.warning(f"Using old .h5 format. Please retrain to use .keras format for better compatibility.")
        else:
            raise FileNotFoundError(f"Model not found. Checked: {model_path_keras} and {model_path_h5}")
        
        self.multi_head_model.load_model(model_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'multi_head_preprocessor.pkl')
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(f"Preprocessor not found: {preprocessor_path}")
        
        with open(preprocessor_path, 'rb') as f:
            preprocessor_data = pickle.load(f)
        
        self.preprocessor.scaler = preprocessor_data['scaler']
        self.preprocessor.sequence_length = preprocessor_data['sequence_length']
        self.preprocessor.scaler_type = preprocessor_data.get('scaler_type', 'minmax')
        self.preprocessor.feature_columns = preprocessor_data.get('feature_columns', MODEL_CONFIG['FEATURE_COLUMNS'])
        self.preprocessor.is_fitted = preprocessor_data.get('is_fitted', True)
        
        self.is_trained = True
        logger.info(f"Multi-head model loaded from {model_path}")
    
    def predict(self, df):
        """
        Make multi-head prediction
        
        Args:
            df: DataFrame with features
        
        Returns:
            Dictionary with all predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Load or train model first.")
        
        # Prepare sequence
        X = self.preprocessor.prepare_for_prediction(df)
        if X is None:
            return None
        
        # Predict
        predictions = self.multi_head_model.predict(X)
        
        # Format results
        result = {
            'trend_continuation': {
                'prediction': int(predictions['trend_continuation'][0] > 0.5),
                'probability': float(predictions['trend_continuation'][0]),
                'confidence': float(max(predictions['trend_continuation'][0], 1 - predictions['trend_continuation'][0]))
            },
            'direction': {
                'prediction': int(predictions['direction'][0] > 0.5),
                'probability': float(predictions['direction'][0]),
                'confidence': float(max(predictions['direction'][0], 1 - predictions['direction'][0]))
            },
            'magnitude': float(predictions['magnitude'][0]),
            'volatility': float(predictions['volatility'][0])
        }
        
        return result


if __name__ == "__main__":
    # Test multi-head trainer
    import pandas as pd
    from config import DATA_CONFIG
    
    training_file = DATA_CONFIG['TRAINING_DATA_FILE']
    if os.path.exists(training_file):
        logger.info(f"Loading training data from {training_file}")
        df = pd.read_csv(training_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Use subset for testing
        df = df.head(2000)
        
        trainer = MultiHeadDLTrainer(sequence_length=60)
        results = trainer.train(df, epochs=5, batch_size=32)
        
        print(f"\nTraining completed!")
        print(f"Trend Accuracy: {results['trend_accuracy']:.4f}")
        print(f"Direction Accuracy: {results['direction_accuracy']:.4f}")
    else:
        print("Training data not found. Run prepare_training_data() first.")

