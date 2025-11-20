"""
Deep Learning Trainer
Handles training pipeline for LSTM models
"""

import os
import numpy as np
import pickle
from logzero import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG, DATA_CONFIG
from models.deep_learning.sequence_preprocessor import SequencePreprocessor
from models.deep_learning.lstm_model import LSTMModel


class DLTrainer:
    """Trains and evaluates deep learning models"""
    
    def __init__(self, sequence_length=60, model_params=None):
        """
        Initialize DL trainer
        
        Args:
            sequence_length: Length of sequences for LSTM
            model_params: Model hyperparameters
        """
        self.sequence_length = sequence_length
        self.preprocessor = SequencePreprocessor(sequence_length=sequence_length)
        self.lstm_model = LSTMModel(sequence_length=sequence_length, model_params=model_params)
        self.model_type = 'lstm'
        self.is_trained = False
    
    def prepare_data(self, df):
        """
        Prepare sequence data for training
        
        Args:
            df: DataFrame with features and target
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        if df is None or df.empty:
            raise ValueError("Empty DataFrame provided")
        
        logger.info(f"Preparing sequence data (sequence_length={self.sequence_length})...")
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(df)
        
        if X is None or y is None:
            raise ValueError("Failed to create sequences")
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=MODEL_CONFIG['TEST_SIZE'],
            random_state=MODEL_CONFIG['RANDOM_STATE'],
            stratify=y
        )
        
        logger.info(f"Train sequences: {len(X_train)}")
        logger.info(f"Test sequences: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, df, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train LSTM model
        
        Args:
            df: Training DataFrame
            epochs: Number of training epochs
            batch_size: Batch size
            validation_split: Validation split ratio
        
        Returns:
            Training history
        """
        logger.info("="*60)
        logger.info("TRAINING LSTM MODEL")
        logger.info("="*60)
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        
        # Calculate class weights for imbalanced data
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        logger.info(f"Class distribution - Class 0 (SELL): {np.sum(y_train == 0)}, Class 1 (BUY): {np.sum(y_train == 1)}")
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Update model params
        self.lstm_model.model_params['epochs'] = epochs
        self.lstm_model.model_params['batch_size'] = batch_size
        self.lstm_model.model_params['validation_split'] = validation_split
        self.lstm_model.model_params['class_weight'] = class_weight_dict
        
        # Build model
        self.lstm_model.build_model()
        
        # Train model with class weights
        history = self.lstm_model.train(
            X_train, y_train,
            X_val=None, y_val=None,  # Use validation_split
            class_weight=class_weight_dict,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        metrics = self.lstm_model.evaluate(X_test, y_test)
        
        # Additional evaluation
        y_pred_proba = self.lstm_model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"\nTest Accuracy: {accuracy:.4f}")
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        logger.info("\nConfusion Matrix:")
        logger.info(confusion_matrix(y_test, y_pred))
        
        self.is_trained = True
        
        return {
            'history': history.history,
            'test_accuracy': accuracy,
            'test_metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
    
    def save_model(self, model_dir=None):
        """
        Save trained model and preprocessor
        
        Args:
            model_dir: Directory to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        if model_dir is None:
            model_dir = MODEL_CONFIG['MODEL_DIR']
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save LSTM model
        lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        self.lstm_model.save_model(lstm_path)
        
        # Save preprocessor (scaler)
        preprocessor_path = os.path.join(model_dir, 'lstm_preprocessor.pkl')
        with open(preprocessor_path, 'wb') as f:
            pickle.dump({
                'scaler': self.preprocessor.scaler,
                'sequence_length': self.preprocessor.sequence_length,
                'scaler_type': self.preprocessor.scaler_type,
                'feature_columns': self.preprocessor.feature_columns,
                'is_fitted': self.preprocessor.is_fitted
            }, f)
        
        logger.info(f"LSTM model saved to {lstm_path}")
        logger.info(f"Preprocessor saved to {preprocessor_path}")
    
    def load_model(self, model_dir=None):
        """
        Load trained model and preprocessor
        
        Args:
            model_dir: Directory containing saved model
        """
        if model_dir is None:
            model_dir = MODEL_CONFIG['MODEL_DIR']
        
        # Load LSTM model
        lstm_path = os.path.join(model_dir, 'lstm_model.h5')
        if not os.path.exists(lstm_path):
            raise FileNotFoundError(f"LSTM model not found: {lstm_path}")
        
        self.lstm_model.load_model(lstm_path)
        
        # Load preprocessor
        preprocessor_path = os.path.join(model_dir, 'lstm_preprocessor.pkl')
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
        logger.info(f"LSTM model loaded from {lstm_path}")
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
    
    def predict(self, df):
        """
        Make prediction on new data
        
        Args:
            df: DataFrame with features
        
        Returns:
            Prediction result dictionary
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Load or train model first.")
        
        # Prepare sequence for prediction
        X = self.preprocessor.prepare_for_prediction(df)
        
        if X is None:
            logger.error("Failed to prepare prediction sequence")
            return None
        
        # Predict
        prediction_proba = self.lstm_model.predict(X)[0]
        prediction = 1 if prediction_proba > 0.5 else 0
        
        return {
            'prediction': prediction,
            'probability': prediction_proba,
            'confidence': max(prediction_proba, 1 - prediction_proba),
            'probabilities': {
                'SELL': 1 - prediction_proba,
                'BUY': prediction_proba
            }
        }


if __name__ == "__main__":
    # Test DL trainer
    import pandas as pd
    from config import DATA_CONFIG
    
    training_file = DATA_CONFIG['TRAINING_DATA_FILE']
    if os.path.exists(training_file):
        logger.info(f"Loading training data from {training_file}")
        df = pd.read_csv(training_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Use subset for testing
        df = df.head(2000)
        
        trainer = DLTrainer(sequence_length=60)
        results = trainer.train(df, epochs=5, batch_size=32)
        
        print(f"\nTraining completed!")
        print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    else:
        print("Training data not found. Run prepare_training_data() first.")

