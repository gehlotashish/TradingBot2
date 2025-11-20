"""
Live Prediction Engine
Makes real-time predictions using trained model
"""

import pandas as pd
import numpy as np
import os
import sys
from logzero import logger
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_CONFIG, SIGNAL_CONFIG, DATA_CONFIG, TRADING_SYMBOLS
from data_pipeline.data_fetcher import DataFetcher
from data_pipeline.data_cleaner import DataCleaner
from feature_engineering.feature_generator import FeatureGenerator
from models.model_trainer import ModelTrainer


class PredictionEngine:
    """Makes live predictions using trained model"""
    
    def __init__(self):
        self.model_trainer = ModelTrainer()
        self.feature_generator = FeatureGenerator()
        self.data_cleaner = DataCleaner()
        self.data_fetcher = DataFetcher(use_live_api=True)
        
        # Load model
        try:
            self.model_trainer.load_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def fetch_live_data(self, symbol_key):
        """
        Fetch recent candles for live prediction
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS
        
        Returns:
            DataFrame with recent OHLCV data
        """
        num_candles = DATA_CONFIG['LIVE_CANDLES']
        df = self.data_fetcher.fetch_live_candles(symbol_key, num_candles)
        
        if df is None or df.empty:
            logger.error(f"Failed to fetch live data for {symbol_key}")
            return None
        
        return df
    
    def prepare_features(self, df):
        """
        Prepare features for prediction
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with features
        """
        # Clean data
        df = self.data_cleaner.clean_ohlcv(df)
        
        # Generate features
        df = self.feature_generator.generate_features(df)
        
        return df
    
    def predict(self, df):
        """
        Make prediction on prepared data
        
        Args:
            df: DataFrame with features
        
        Returns:
            Dictionary with prediction results
        """
        if df is None or df.empty:
            logger.error("Empty DataFrame for prediction")
            return None
        
        # Get last row (most recent candle)
        last_row = df.iloc[[-1]]
        
        # Select feature columns
        feature_cols = MODEL_CONFIG['FEATURE_COLUMNS']
        missing_cols = [col for col in feature_cols if col not in last_row.columns]
        
        if missing_cols:
            logger.error(f"Missing feature columns: {missing_cols}")
            return None
        
        X = last_row[feature_cols]
        
        # Check for NaN
        if X.isna().any().any():
            logger.warning("NaN values in features, filling with 0")
            X = X.fillna(0)
        
        # Make prediction
        try:
            prediction = self.model_trainer.model.predict(X)[0]
            probabilities = self.model_trainer.model.predict_proba(X)[0]
            
            # Get confidence (probability of predicted class)
            confidence = float(probabilities[prediction])
            
            # Get signal
            signal = "BUY" if prediction == 1 else "SELL"
            
            # Get current price
            current_price = float(last_row['close'].iloc[0])
            
            # Calculate trigger price
            atr = float(last_row['atr'].iloc[0]) if 'atr' in last_row.columns else 0
            if signal == "BUY":
                trigger_price = current_price + (atr * SIGNAL_CONFIG['BREAKOUT_MULTIPLIER'])
            else:
                trigger_price = current_price - (atr * SIGNAL_CONFIG['BREAKDOWN_MULTIPLIER'])
            
            # Get reason (based on indicators)
            reason = self._generate_reason(last_row)
            
            result = {
                'timestamp': datetime.now(),
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'trigger_price': trigger_price,
                'prediction': int(prediction),
                'probabilities': {
                    'SELL': float(probabilities[0]),
                    'BUY': float(probabilities[1])
                },
                'reason': reason,
                'indicators': {
                    'rsi': float(last_row['rsi'].iloc[0]) if 'rsi' in last_row.columns else None,
                    'macd': float(last_row['macd'].iloc[0]) if 'macd' in last_row.columns else None,
                    'ema_9': float(last_row['ema_9'].iloc[0]) if 'ema_9' in last_row.columns else None,
                    'ema_21': float(last_row['ema_21'].iloc[0]) if 'ema_21' in last_row.columns else None,
                }
            }
            
            logger.info(f"Prediction: {signal} | Confidence: {confidence:.2%} | Price: {current_price:.2f}")
            
            return result
            
        except Exception as e:
            logger.exception(f"Error making prediction: {e}")
            return None
    
    def _generate_reason(self, last_row):
        """Generate human-readable reason for signal"""
        reasons = []
        
        if 'rsi' in last_row.columns:
            rsi = float(last_row['rsi'].iloc[0])
            if rsi < 30:
                reasons.append("RSI oversold")
            elif rsi > 70:
                reasons.append("RSI overbought")
        
        if 'macd' in last_row.columns and 'macd_signal' in last_row.columns:
            macd = float(last_row['macd'].iloc[0])
            macd_signal = float(last_row['macd_signal'].iloc[0])
            if macd > macd_signal:
                reasons.append("MACD bullish")
            else:
                reasons.append("MACD bearish")
        
        if 'ema_9' in last_row.columns and 'ema_21' in last_row.columns:
            ema_9 = float(last_row['ema_9'].iloc[0])
            ema_21 = float(last_row['ema_21'].iloc[0])
            if ema_9 > ema_21:
                reasons.append("EMA bullish crossover")
            else:
                reasons.append("EMA bearish crossover")
        
        if 'volume_ratio' in last_row.columns:
            vol_ratio = float(last_row['volume_ratio'].iloc[0])
            if vol_ratio > 1.5:
                reasons.append("High volume")
        
        return ", ".join(reasons) if reasons else "Technical analysis"
    
    def generate_signal(self, symbol_key):
        """
        Complete pipeline: fetch data, prepare features, predict
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS
        
        Returns:
            Dictionary with signal information
        """
        # Connect if not connected
        if not self.data_fetcher.connected:
            if not self.data_fetcher.connect():
                logger.error("Failed to connect to SmartAPI")
                return None
        
        # Fetch live data
        df = self.fetch_live_data(symbol_key)
        if df is None:
            return None
        
        # Prepare features
        df = self.prepare_features(df)
        if df is None or df.empty:
            return None
        
        # Make prediction
        result = self.predict(df)
        
        if result:
            # Add symbol information
            symbol_info = TRADING_SYMBOLS[symbol_key]
            result['symbol'] = symbol_info['name']
            result['symbol_key'] = symbol_key
            result['exchange'] = symbol_info['exchange']
        
        return result


if __name__ == "__main__":
    # Test prediction engine
    engine = PredictionEngine()
    
    # Test prediction for BANKNIFTY
    result = engine.generate_signal('BANKNIFTY')
    
    if result:
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Symbol: {result['symbol']}")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Current Price: ₹{result['current_price']:.2f}")
        print(f"Trigger Price: ₹{result['trigger_price']:.2f}")
        print(f"Reason: {result['reason']}")
        print("="*50)

