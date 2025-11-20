"""
Multi-Head LSTM Model
Predicts multiple targets simultaneously: trend, magnitude, volatility, direction
Hedge fund-style architecture
"""

import numpy as np
from logzero import logger
import sys
import os

# Try to import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    try:
        import keras
        from keras.models import Model
        from keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Concatenate
        from keras.optimizers import Adam
        from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        logger.warning("TensorFlow/Keras not available. Install with: pip install tensorflow")
        TENSORFLOW_AVAILABLE = False

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import MODEL_CONFIG, LSTM_PARAMS


class MultiHeadLSTMModel:
    """Multi-head LSTM model for simultaneous prediction of multiple targets"""
    
    def __init__(self, sequence_length=60, num_features=None, model_params=None):
        """
        Initialize Multi-Head LSTM model
        
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
            'lstm_units_3': 32,
            'dense_units': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.0005,
            'batch_size': 64,
            'epochs': 100,
            'validation_split': 0.2,
        }
        
        self.model_params = {**default_params, **(model_params or {})}
    
    def build_model(self):
        """Build multi-head LSTM model architecture"""
        logger.info("Building Multi-Head LSTM model architecture...")
        
        from tensorflow.keras.initializers import GlorotUniform
        
        # Input layer
        input_layer = Input(shape=(self.sequence_length, self.num_features), name='input')
        
        # Shared LSTM layers (feature extraction)
        x = LSTM(
            units=self.model_params['lstm_units_1'],
            return_sequences=True,
            kernel_initializer=GlorotUniform(seed=42),
            recurrent_initializer=GlorotUniform(seed=42),
            name='lstm_1'
        )(input_layer)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(self.model_params['dropout_rate'], name='dropout_1')(x)
        
        x = LSTM(
            units=self.model_params['lstm_units_2'],
            return_sequences=True,
            kernel_initializer=GlorotUniform(seed=42),
            recurrent_initializer=GlorotUniform(seed=42),
            name='lstm_2'
        )(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(self.model_params['dropout_rate'], name='dropout_2')(x)
        
        x = LSTM(
            units=self.model_params['lstm_units_3'],
            return_sequences=False,
            kernel_initializer=GlorotUniform(seed=42),
            recurrent_initializer=GlorotUniform(seed=42),
            name='lstm_3'
        )(x)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(self.model_params['dropout_rate'], name='dropout_3')(x)
        
        # Shared dense layer
        shared_dense = Dense(
            self.model_params['dense_units'],
            activation='relu',
            kernel_initializer=GlorotUniform(seed=42),
            name='shared_dense'
        )(x)
        shared_dense = BatchNormalization(name='bn_4')(shared_dense)
        shared_dense = Dropout(self.model_params['dropout_rate'] * 0.5, name='dropout_4')(shared_dense)
        
        # Multi-head outputs
        
        # Head 1: Trend Continuation (Binary Classification)
        trend_branch = Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42), name='trend_dense')(shared_dense)
        trend_branch = Dropout(0.2, name='trend_dropout')(trend_branch)
        trend_output = Dense(1, activation='sigmoid', name='trend_continuation')(trend_branch)
        
        # Head 2: Direction (Binary Classification)
        direction_branch = Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42), name='direction_dense')(shared_dense)
        direction_branch = Dropout(0.2, name='direction_dropout')(direction_branch)
        direction_output = Dense(1, activation='sigmoid', name='direction')(direction_branch)
        
        # Head 3: Magnitude (Regression)
        magnitude_branch = Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42), name='magnitude_dense')(shared_dense)
        magnitude_branch = Dropout(0.2, name='magnitude_dropout')(magnitude_branch)
        magnitude_output = Dense(1, activation='linear', name='magnitude')(magnitude_branch)
        
        # Head 4: Volatility (Regression)
        volatility_branch = Dense(32, activation='relu', kernel_initializer=GlorotUniform(seed=42), name='volatility_dense')(shared_dense)
        volatility_branch = Dropout(0.2, name='volatility_dropout')(volatility_branch)
        volatility_output = Dense(1, activation='linear', name='volatility')(volatility_branch)
        
        # Create model with multiple outputs
        self.model = Model(
            inputs=input_layer,
            outputs={
                'trend_continuation': trend_output,
                'direction': direction_output,
                'magnitude': magnitude_output,
                'volatility': volatility_output
            },
            name='multi_head_lstm'
        )
        
        # Compile with multiple losses
        optimizer = Adam(
            learning_rate=self.model_params.get('learning_rate', 0.0005),
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        # Different losses for different heads
        # Use proper loss objects instead of strings for better compatibility
        from keras.losses import MeanSquaredError, BinaryCrossentropy
        
        losses = {
            'trend_continuation': BinaryCrossentropy(),
            'direction': BinaryCrossentropy(),
            'magnitude': MeanSquaredError(),  # Mean Squared Error for regression
            'volatility': MeanSquaredError()   # Mean Squared Error for regression
        }
        
        # Loss weights (optimized for trading signals)
        # Direction is the main signal (BUY/SELL) - highest weight
        # Trend is weak (51% accuracy) - lower weight
        # Magnitude and Volatility are important for position sizing and stop-loss
        loss_weights = {
            'trend_continuation': 0.2,  # Reduced: weak head (51% accuracy)
            'direction': 1.5,           # Increased: main signal (75.7% accuracy, needs more focus)
            'magnitude': 0.7,            # Increased: important for position sizing
            'volatility': 0.3           # Maintained: important for stop-loss
        }
        
        # Metrics for each head (use proper metric objects for better compatibility)
        from keras.metrics import MeanAbsoluteError, Accuracy, Precision, Recall
        
        metrics = {
            'trend_continuation': [Accuracy(), Precision(), Recall()],
            'direction': [Accuracy(), Precision(), Recall()],
            'magnitude': [MeanAbsoluteError()],  # Mean Absolute Error
            'volatility': [MeanAbsoluteError()]
        }
        
        self.model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        logger.info("Multi-Head LSTM model built successfully")
        logger.info(f"Model parameters: {self.model.count_params():,} trainable parameters")
        logger.info("Outputs: trend_continuation, direction, magnitude, volatility")
        
        return self.model
    
    def get_callbacks(self, model_path=None, patience=15):
        """Get training callbacks"""
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
    
    def train(self, X_train, y_train_dict, X_val=None, y_val_dict=None, class_weight=None, verbose=1):
        """
        Train multi-head model
        
        Args:
            X_train: Training sequences
            y_train_dict: Dictionary with targets {
                'trend_continuation': ...,
                'direction': ...,
                'magnitude': ...,
                'volatility': ...
            }
            X_val: Validation sequences
            y_val_dict: Validation targets
            class_weight: Class weights for classification heads
            verbose: Verbosity level
        
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        logger.info("Starting Multi-Head LSTM model training...")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Sequence shape: {X_train.shape}")
        
        # Verify data
        for key, y in y_train_dict.items():
            if np.all(y == y[0]) if len(np.unique(y)) == 1 else False:
                logger.warning(f"All {key} targets are the same!")
        
        # Check for NaN/Inf
        if np.isnan(X_train).any() or np.isinf(X_train).any():
            logger.warning("NaN or Inf values in X_train, replacing with 0")
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val_dict is not None:
            validation_data = (X_val, y_val_dict)
        
        # Prepare class weights for classification heads
        sample_weight = None
        if class_weight:
            # Create sample weights for classification heads
            trend_weights = np.array([class_weight.get(0, 1.0) if y == 0 else class_weight.get(1, 1.0) 
                                      for y in y_train_dict['trend_continuation']])
            direction_weights = np.array([class_weight.get(0, 1.0) if y == 0 else class_weight.get(1, 1.0) 
                                         for y in y_train_dict['direction']])
            
            sample_weight = {
                'trend_continuation': trend_weights,
                'direction': direction_weights,
                'magnitude': np.ones(len(X_train)),  # No weights for regression
                'volatility': np.ones(len(X_train))
            }
        
        # Get callbacks
        callbacks = self.get_callbacks(patience=15)
        
        # Train model
        history = self.model.fit(
            X_train, y_train_dict,
            batch_size=self.model_params['batch_size'],
            epochs=self.model_params['epochs'],
            validation_data=validation_data,
            validation_split=self.model_params.get('validation_split', 0.2) if validation_data is None else None,
            sample_weight=sample_weight,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=True
        )
        
        logger.info("Multi-Head LSTM model training completed")
        
        return history
    
    def predict(self, X):
        """
        Make predictions on all heads
        
        Args:
            X: Input sequences
        
        Returns:
            Dictionary with predictions from all heads
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        
        return {
            'trend_continuation': predictions['trend_continuation'].flatten(),
            'direction': predictions['direction'].flatten(),
            'magnitude': predictions['magnitude'].flatten(),
            'volatility': predictions['volatility'].flatten()
        }
    
    def evaluate(self, X_test, y_test_dict):
        """Evaluate model on all heads"""
        if self.model is None:
            raise ValueError("Model not trained.")
        
        results = self.model.evaluate(X_test, y_test_dict, verbose=0)
        
        # Parse results (Keras returns flat list)
        metric_names = self.model.metrics_names
        metrics = dict(zip(metric_names, results))
        
        logger.info("Multi-Head Model Evaluation:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def save_model(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(filepath)
        logger.info(f"Multi-Head LSTM model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load model from file"""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow required to load model")
        
        is_old_format = filepath.endswith('.h5')
        
        try:
            # Use safe_mode=False for custom losses/metrics
            self.model = keras.models.load_model(filepath, safe_mode=False)
            logger.info(f"Multi-Head LSTM model loaded from {filepath}")
        except Exception as e:
            # If loading fails due to old format with string losses/metrics, recompile
            if is_old_format and ('mse' in str(e).lower() or 'mae' in str(e).lower()):
                logger.warning(f"Old .h5 format detected with string losses/metrics. Recompiling with proper objects...")
                # Load model architecture only (without compilation)
                self.model = keras.models.load_model(filepath, compile=False, safe_mode=False)
                
                # Recompile with proper loss/metric objects
                from keras.losses import MeanSquaredError, BinaryCrossentropy
                from keras.metrics import MeanAbsoluteError, Accuracy, Precision, Recall
                from keras.optimizers import Adam
                
                optimizer = Adam(
                    learning_rate=self.model_params.get('learning_rate', 0.0005),
                    clipnorm=1.0,
                    beta_1=0.9,
                    beta_2=0.999,
                    epsilon=1e-7
                )
                
                losses = {
                    'trend_continuation': BinaryCrossentropy(),
                    'direction': BinaryCrossentropy(),
                    'magnitude': MeanSquaredError(),
                    'volatility': MeanSquaredError()
                }
                
                loss_weights = {
                    'trend_continuation': 0.2,
                    'direction': 1.5,
                    'magnitude': 0.7,
                    'volatility': 0.3
                }
                
                metrics = {
                    'trend_continuation': [Accuracy(), Precision(), Recall()],
                    'direction': [Accuracy(), Precision(), Recall()],
                    'magnitude': [MeanAbsoluteError()],
                    'volatility': [MeanAbsoluteError()]
                }
                
                self.model.compile(
                    optimizer=optimizer,
                    loss=losses,
                    loss_weights=loss_weights,
                    metrics=metrics
                )
                logger.info(f"Model recompiled with proper loss/metric objects")
            else:
                raise
        
        # Update sequence_length and num_features from loaded model
        input_shape = self.model.input_shape
        if input_shape:
            self.sequence_length = input_shape[1] if len(input_shape) > 1 else None
            self.num_features = input_shape[2] if len(input_shape) > 2 else None


if __name__ == "__main__":
    # Test multi-head model
    print("Testing Multi-Head LSTM model architecture...")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Install with: pip install tensorflow")
    else:
        model = MultiHeadLSTMModel(sequence_length=60, num_features=27)
        model.build_model()
        print("\nModel Summary:")
        model.model.summary()

