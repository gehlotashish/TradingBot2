"""
Ensemble Model
Combines predictions from multiple models (LSTM, RandomForest, XGBoost)
"""

import numpy as np
from logzero import logger
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DL_CONFIG


class EnsembleModel:
    """Ensembles predictions from multiple models"""
    
    def __init__(self):
        self.models = {}
        self.weights = DL_CONFIG.get('ENSEMBLE_WEIGHTS', {
            'lstm': 0.4,
            'random_forest': 0.3,
            'xgboost': 0.3
        })
        self.use_ensemble = DL_CONFIG.get('USE_ENSEMBLE', True)
    
    def add_model(self, model_name, model, model_type='traditional'):
        """
        Add a model to the ensemble
        
        Args:
            model_name: Name of the model ('lstm', 'random_forest', 'xgboost')
            model: Model object
            model_type: 'lstm' or 'traditional'
        """
        self.models[model_name] = {
            'model': model,
            'type': model_type
        }
        logger.info(f"Added {model_name} to ensemble")
    
    def predict(self, X_traditional=None, X_sequence=None):
        """
        Make ensemble prediction
        
        Args:
            X_traditional: Input for traditional ML models (DataFrame or array)
            X_sequence: Input sequence for LSTM (array)
        
        Returns:
            Dictionary with ensemble prediction and individual predictions
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        predictions = {}
        probabilities = {}
        
        # Get predictions from each model
        for model_name, model_info in self.models.items():
            model = model_info['model']
            model_type = model_info['type']
            
            try:
                if model_type == 'lstm':
                    if X_sequence is None:
                        logger.warning(f"Skipping {model_name}: No sequence data provided")
                        continue
                    # LSTM prediction
                    pred_result = model.predict(X_sequence)
                    if isinstance(pred_result, dict):
                        predictions[model_name] = pred_result['prediction']
                        probabilities[model_name] = pred_result['probabilities']
                    else:
                        # Assume it's a probability
                        prob = pred_result if isinstance(pred_result, (int, float)) else pred_result[0]
                        predictions[model_name] = 1 if prob > 0.5 else 0
                        probabilities[model_name] = {'SELL': 1 - prob, 'BUY': prob}
                
                else:  # traditional ML
                    if X_traditional is None:
                        logger.warning(f"Skipping {model_name}: No traditional data provided")
                        continue
                    # Traditional ML prediction
                    pred = model.predict(X_traditional)[0] if hasattr(model, 'predict') else None
                    proba = model.predict_proba(X_traditional)[0] if hasattr(model, 'predict_proba') else None
                    
                    if pred is not None:
                        predictions[model_name] = int(pred)
                    if proba is not None:
                        probabilities[model_name] = {
                            'SELL': float(proba[0]),
                            'BUY': float(proba[1])
                        }
            except Exception as e:
                logger.exception(f"Error getting prediction from {model_name}: {e}")
                continue
        
        if not predictions:
            logger.error("No valid predictions from any model")
            return None
        
        # Weighted ensemble prediction
        if self.use_ensemble and len(predictions) > 1:
            ensemble_result = self._weighted_ensemble(probabilities)
        else:
            # Use single model if only one available
            model_name = list(predictions.keys())[0]
            ensemble_result = {
                'prediction': predictions[model_name],
                'probabilities': probabilities[model_name],
                'confidence': max(probabilities[model_name].values())
            }
        
        ensemble_result['individual_predictions'] = predictions
        ensemble_result['individual_probabilities'] = probabilities
        
        return ensemble_result
    
    def _weighted_ensemble(self, probabilities):
        """
        Calculate weighted ensemble prediction
        
        Args:
            probabilities: Dictionary of probabilities from each model
        
        Returns:
            Ensemble prediction result
        """
        # Normalize weights
        total_weight = sum(self.weights.get(name, 0) for name in probabilities.keys())
        if total_weight == 0:
            # Equal weights if not specified
            weight_per_model = 1.0 / len(probabilities)
            weights = {name: weight_per_model for name in probabilities.keys()}
        else:
            weights = {name: self.weights.get(name, 0) / total_weight for name in probabilities.keys()}
        
        # Weighted average of probabilities
        ensemble_buy_prob = sum(
            probabilities[model_name]['BUY'] * weights[model_name]
            for model_name in probabilities.keys()
        )
        
        ensemble_sell_prob = 1 - ensemble_buy_prob
        
        # Final prediction
        prediction = 1 if ensemble_buy_prob > 0.5 else 0
        confidence = max(ensemble_buy_prob, ensemble_sell_prob)
        
        return {
            'prediction': prediction,
            'probabilities': {
                'SELL': ensemble_sell_prob,
                'BUY': ensemble_buy_prob
            },
            'confidence': confidence,
            'weights_used': weights
        }
    
    def get_model_count(self):
        """Get number of models in ensemble"""
        return len(self.models)
    
    def is_ready(self):
        """Check if ensemble is ready (has at least one model)"""
        return len(self.models) > 0


if __name__ == "__main__":
    # Test ensemble
    print("Ensemble Model Test")
    ensemble = EnsembleModel()
    print(f"Models in ensemble: {ensemble.get_model_count()}")
    print(f"Ensemble ready: {ensemble.is_ready()}")

