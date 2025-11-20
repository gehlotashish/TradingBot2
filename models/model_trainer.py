"""
Model Training Module
Trains ML models (RandomForest, XGBoost) on prepared data
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from logzero import logger
import sys

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    logger.warning("XGBoost not available. Install with: pip install xgboost")
    XGBOOST_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, RF_PARAMS, XGB_PARAMS, DATA_CONFIG


class ModelTrainer:
    """Trains and evaluates ML models"""
    
    def __init__(self):
        self.model = None
        self.model_type = None
        self.feature_columns = MODEL_CONFIG['FEATURE_COLUMNS']
        self.target_column = MODEL_CONFIG['TARGET_COLUMN']
        self.test_size = MODEL_CONFIG['TEST_SIZE']
        self.random_state = MODEL_CONFIG['RANDOM_STATE']
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Select feature columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = df[self.feature_columns].copy()
        y = df[self.target_column].copy()
        
        # Remove rows with NaN in target
        valid_mask = ~y.isna()
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Remove rows with NaN in features
        valid_mask = ~X.isna().any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Prepared data: {len(X)} samples, {len(self.feature_columns)} features")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest model...")
        
        self.model = RandomForestClassifier(**RF_PARAMS)
        self.model.fit(X_train, y_train)
        self.model_type = 'random_forest'
        
        logger.info("Random Forest model trained successfully")
        return self.model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost model"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: pip install xgboost")
        
        logger.info("Training XGBoost model...")
        
        self.model = xgb.XGBClassifier(**XGB_PARAMS)
        self.model.fit(X_train, y_train)
        self.model_type = 'xgboost'
        
        logger.info("XGBoost model trained successfully")
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info(f"\nModel Evaluation ({self.model_type}):")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info(f"\nConfusion Matrix:")
        logger.info(cm)
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            logger.info("\nTop 10 Feature Importances:")
            logger.info(feature_importance.head(10).to_string())
            
            return feature_importance
        else:
            logger.warning("Model does not support feature importance")
            return None
    
    def save_model(self, filepath=None):
        """Save trained model to file"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if filepath is None:
            filepath = MODEL_CONFIG['MODEL_FILE']
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'model_type': self.model_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column
            }, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath=None):
        """Load trained model from file"""
        if filepath is None:
            filepath = MODEL_CONFIG['MODEL_FILE']
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_columns = model_data.get('feature_columns', self.feature_columns)
        self.target_column = model_data.get('target_column', self.target_column)
        
        logger.info(f"Model loaded from {filepath}")
        return self.model
    
    def train_and_save(self, df, model_type='random_forest'):
        """
        Complete training pipeline: prepare data, train, evaluate, save
        
        Args:
            df: DataFrame with features and target
            model_type: 'random_forest' or 'xgboost'
        
        Returns:
            Evaluation metrics
        """
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Train model
        if model_type == 'random_forest':
            self.train_random_forest(X_train, y_train)
        elif model_type == 'xgboost':
            self.train_xgboost(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate
        metrics = self.evaluate_model(X_test, y_test)
        
        # Feature importance
        self.get_feature_importance()
        
        # Save model
        self.save_model()
        
        return metrics


if __name__ == "__main__":
    # Test model trainer
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_pipeline.data_fetcher import DataFetcher
    from data_pipeline.data_cleaner import DataCleaner
    from feature_engineering.feature_generator import FeatureGenerator
    from models.target_generator import TargetGenerator
    
    # Load or prepare data
    training_file = DATA_CONFIG['TRAINING_DATA_FILE']
    
    if os.path.exists(training_file):
        logger.info(f"Loading training data from {training_file}")
        df = pd.read_csv(training_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        logger.info("Fetching and preparing training data...")
        fetcher = DataFetcher()
        if not fetcher.connect():
            logger.error("Failed to connect to SmartAPI")
            exit(1)
        
        # Fetch data for both symbols
        df_banknifty = fetcher.fetch_training_data('BANKNIFTY')
        df_nifty = fetcher.fetch_training_data('NIFTY50')
        
        if df_banknifty is None or df_nifty is None:
            logger.error("Failed to fetch data")
            exit(1)
        
        # Clean and generate features
        cleaner = DataCleaner()
        generator = FeatureGenerator()
        target_gen = TargetGenerator()
        
        df_banknifty = cleaner.clean_ohlcv(df_banknifty)
        df_banknifty = generator.generate_features(df_banknifty)
        df_banknifty['target'] = target_gen.generate_binary_target(df_banknifty)
        df_banknifty['symbol'] = 'BANKNIFTY'
        
        df_nifty = cleaner.clean_ohlcv(df_nifty)
        df_nifty = generator.generate_features(df_nifty)
        df_nifty['target'] = target_gen.generate_binary_target(df_nifty)
        df_nifty['symbol'] = 'NIFTY50'
        
        # Combine
        df = pd.concat([df_banknifty, df_nifty], ignore_index=True)
        df = df.dropna().reset_index(drop=True)
    
    # Train model
    trainer = ModelTrainer()
    metrics = trainer.train_and_save(df, model_type='random_forest')
    
    print(f"\n✓ Model training completed!")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

