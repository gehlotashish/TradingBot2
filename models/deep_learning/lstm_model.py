"""
LSTM Model Architecture for Trading Signal Prediction
"""

import numpy as np
from logzero import logger
import sys
import os

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras.models import Sequential
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        logger.warning("TensorFlow/Keras not available. Install with: pip install tensorflow")
        TENSORFLOW_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG


class LSTMModel:
    """LSTM model for trading signal prediction"""
    
    def __init__(self, sequence_length=60, num_features=None, model_params=None):
        """
        Initialize LSTM model
        
        Args:
            sequence_length: Length of input sequences
            num_features: Number of features per time step
            model_params: Dictionary with model hyperparameters
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow/Keras is required. Install with: pip install tensorflow")
        
        self.sequence_length = sequence_length
        self.num_features = num_features or len(MODEL_CONFIG['FEATURE_COLUMNS'])
        self.model = None
        
        # Default model parameters
        default_params = {
            'lstm_units_1': 128,
            'lstm_units_2': 64,
            'dense_units': 32,
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'validation_split': 0.2,
        }
        
        self.model_params = {**default_params, **(model_params or {})}
    
    def build_model(self):
        """Build LSTM model architecture"""
        logger.info("Building LSTM model architecture...")
        
        # Use better initialization
        from tensorflow.keras.initializers import GlorotUniform
        
        # Build deeper, more powerful model
        model = Sequential([
            # First LSTM layer - larger capacity
            LSTM(
                units=self.model_params['lstm_units_1'],
                return_sequences=True,
                input_shape=(self.sequence_length, self.num_features),
                kernel_initializer=GlorotUniform(seed=42),
                recurrent_initializer=GlorotUniform(seed=42),
                name='lstm_1'
            ),
            BatchNormalization(name='bn_1'),
            Dropout(self.model_params['dropout_rate'], name='dropout_1'),
            
            # Second LSTM layer
            LSTM(
                units=self.model_params['lstm_units_2'],
                return_sequences=True,  # Changed to True for third layer
                kernel_initializer=GlorotUniform(seed=42),
                recurrent_initializer=GlorotUniform(seed=42),
                name='lstm_2'
            ),
            BatchNormalization(name='bn_2'),
            Dropout(self.model_params['dropout_rate'], name='dropout_2'),
            
            # Third LSTM layer - added for more capacity
            LSTM(
                units=self.model_params['lstm_units_2'] // 2,  # 32 units
                return_sequences=False,
                kernel_initializer=GlorotUniform(seed=42),
                recurrent_initializer=GlorotUniform(seed=42),
                name='lstm_3'
            ),
            BatchNormalization(name='bn_3'),
            Dropout(self.model_params['dropout_rate'], name='dropout_3'),
            
            # Dense layers with better initialization
            Dense(
                self.model_params['dense_units'] * 2,  # 64 units
                activation='relu',
                kernel_initializer=GlorotUniform(seed=42),
                name='dense_1'
            ),
            BatchNormalization(name='bn_4'),
            Dropout(self.model_params['dropout_rate'] * 0.5, name='dropout_4'),
            
            # Second dense layer
            Dense(
                self.model_params['dense_units'],  # 32 units
                activation='relu',
                kernel_initializer=GlorotUniform(seed=42),
                name='dense_2'
            ),
            BatchNormalization(name='bn_5'),
            Dropout(self.model_params['dropout_rate'] * 0.3, name='dropout_5'),
            
            # Output layer (binary classification: BUY/SELL)
            Dense(
                1,
                activation='sigmoid',
                kernel_initializer=GlorotUniform(seed=42),
                name='output'
            )
        ])
        
        # Compile model with learning rate and gradient clipping
        # Use learning rate schedule
        initial_lr = self.model_params.get('learning_rate', 0.0005)
        optimizer = Adam(
            learning_rate=initial_lr,
            clipnorm=1.0,  # Gradient clipping to prevent exploding gradients
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.model = model
        
        logger.info("LSTM model built successfully")
        logger.info(f"Model parameters: {model.count_params():,} trainable parameters")
        
        return model
    
    def get_callbacks(self, model_path=None, patience=10):
        """
        Get training callbacks
        
        Args:
            model_path: Path to save best model
            patience: Early stopping patience
        
        Returns:
            List of callbacks
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        if model_path:
            callbacks.append(
                ModelCheckpoint(
                    model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        return callbacks
    
    def train(self, X_train, y_train, X_val=None, y_val=None, class_weight=None, verbose=1):
        """
        Train LSTM model
        
        Args:
            X_train: Training sequences (samples, sequence_length, features)
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            class_weight: Class weights dictionary for imbalanced data
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Starting LSTM model training...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Sequence shape: {X_train.shape}")
        logger.info(f"Target distribution - Class 0: {np.sum(y_train == 0)}, Class 1: {np.sum(y_train == 1)}")
        
        # Verify data
        if np.all(y_train == 0) or np.all(y_train == 1):
            logger.error("All targets are the same class! Check target generation.")
            raise ValueError("Invalid target distribution")
        
        # Check for NaN/Inf
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            logger.warning("NaN or Inf values in X_train, replacing with 0")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        elif self.model_params.get('validation_split', 0) > 0:
            # Use validation_split
            pass
        
        # Get callbacks
        callbacks = self.get_callbacks(patience=10)
        
        # Train model with class weights
        history = self.model.fit(
            X_train, y_train,
            batch_size=self.model_params['batch_size'],
            epochs=self.model_params['epochs'],
            validation_data=validation_data,
            validation_split=self.model_params.get('validation_split', 0.2) if validation_data is None else None,
            class_weight=class_weight,  # Add class weights
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        logger.info("LSTM model training completed")
        
        return history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences (samples, sequence_length, features)
        
        Returns:
            Predictions (probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def predict_proba(self, X):
        """
        Predict probabilities for both classes
        
        Args:
            X: Input sequences
        
        Returns:
            Array of [SELL_prob, BUY_prob] for each sample
        """
        predictions = self.predict(X)
        # Convert to [SELL_prob, BUY_prob] format
        proba = np.column_stack([1 - predictions, predictions])
        return proba
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model
        
        Args:
            X_test: Test sequences
            y_test: Test targets
        
        Returns:
            Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {
            'loss': results[0],
            'accuracy': results[1] if len(results) > 1 else None,
            'precision': results[2] if len(results) > 2 else None,
            'recall': results[3] if len(results) > 3 else None
        }
        
        logger.info(f"LSTM Model Evaluation:")
        logger.info(f"  Loss: {metrics['loss']:.4f}")
        if metrics['accuracy']:
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        if metrics['precision']:
            logger.info(f"  Precision: {metrics['precision']:.4f}")
        if metrics['recall']:
            logger.info(f"  Recall: {metrics['recall']:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"LSTM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required to load model")
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"LSTM model loaded from {filepath}")
        
        # Update sequence_length and num_features from loaded model
        input_shape = self.model.input_shape
        if input_shape:
            self.sequence_length = input_shape[1] if len(input_shape) > 1 else None
            self.num_features = input_shape[2] if len(input_shape) > 2 else None
    
    def get_model_summary(self):
        """Get model summary"""
        if self.model is None:
            return "Model not built yet"
        return self.model.summary()


if __name__ == "__main__":
    # Test LSTM model
    print("Testing LSTM model architecture...")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Install with: pip install tensorflow")
    else:
        model = LSTMModel(sequence_length=60, num_features=20)
        model.build_model()
        print("\nModel Summary:")
        model.model.summary()

