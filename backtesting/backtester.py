"""
Backtesting Module
Tests model performance on historical data
"""

import pandas as pd
import numpy as np
from logzero import logger
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from live_engine.prediction_engine import PredictionEngine
from config import DATA_CONFIG, TRADING_SYMBOLS


class Backtester:
    """Backtests trading signals on historical data"""
    
    def __init__(self):
        """Initialize backtester"""
        self.engine = PredictionEngine()
        self.results = []
        self.trades = []
    
    def run(self, test_data_file=None, symbol='BANKNIFTY', max_candles=None):
        """
        Run backtest on historical data
        
        Args:
            test_data_file: Path to test data CSV (default: training_ready_data.csv)
            symbol: Trading symbol to backtest
            max_candles: Maximum number of candles to process (None for all)
        """
        logger.info("="*60)
        logger.info("STARTING BACKTEST")
        logger.info("="*60)
        
        # Load test data
        if test_data_file is None:
            test_data_file = DATA_CONFIG['TRAINING_DATA_FILE']
        
        if not os.path.exists(test_data_file):
            logger.error(f"Test data file not found: {test_data_file}")
            logger.info("Using training data for backtest...")
            test_data_file = DATA_CONFIG['TRAINING_DATA_FILE']
        
        if not os.path.exists(test_data_file):
            logger.error("No data file found for backtesting")
            logger.info("Please run: python main.py --prepare-data")
            return
        
        logger.info(f"Loading test data from: {test_data_file}")
        df = pd.read_csv(test_data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Filter by symbol if available
        if 'symbol' in df.columns:
            df = df[df['symbol'] == symbol].copy()
            logger.info(f"Filtered data for {symbol}: {len(df)} rows")
        
        if df.empty:
            logger.error("No data available for backtesting")
            return
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Limit candles if specified (for faster testing)
        if max_candles and len(df) > max_candles:
            df = df.tail(max_candles).reset_index(drop=True)
            logger.info(f"Limited to {max_candles} most recent candles")
        
        logger.info(f"Backtesting on {len(df)} candles")
        logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        # Run backtest
        self._run_backtest(df, symbol)
        
        # Calculate metrics
        self._calculate_metrics()
        
        # Print results
        self._print_results()
    
    def _run_backtest(self, df, symbol):
        """Run backtest on dataframe"""
        logger.info("Running predictions...")
        
        # We need at least sequence_length candles for LSTM
        sequence_length = 60
        min_candles = sequence_length + 1
        
        if len(df) < min_candles:
            logger.error(f"Insufficient data: {len(df)} < {min_candles}")
            return
        
        # Process in batches (simulate live prediction)
        # For backtesting, we'll process each candle sequentially
        # but need enough history for sequences
        
        total_candles = len(df) - min_candles
        progress_interval = max(1, total_candles // 20)  # Show progress every 5%
        
        for i in range(min_candles, len(df)):
            # Show progress
            if (i - min_candles) % progress_interval == 0:
                progress = ((i - min_candles) / total_candles) * 100
                logger.info(f"Progress: {progress:.1f}% ({i - min_candles}/{total_candles} candles processed)")
            # Get data up to current point (simulating what we'd have at that time)
            current_data = df.iloc[:i+1].copy()
            
            try:
                # Prepare features first
                prepared_data = self.engine.prepare_features(current_data)
                
                if prepared_data is None or prepared_data.empty:
                    continue
                
                # Make prediction
                prediction = self.engine.predict(prepared_data)
                
                if prediction:
                    # Store result
                    result = {
                        'timestamp': current_data['timestamp'].iloc[-1],
                        'price': current_data['close'].iloc[-1],
                        'signal': prediction['signal'],
                        'confidence': prediction['confidence'],
                        'prediction': prediction['prediction'],
                        'model_type': prediction.get('model_type', 'unknown')
                    }
                    
                    # Add actual outcome (if target available)
                    if 'target' in current_data.columns and i < len(df) - 1:
                        actual = df.iloc[i+1]['target'] if i+1 < len(df) else None
                        result['actual'] = actual
                        result['correct'] = (prediction['prediction'] == actual) if actual is not None else None
                    
                    self.results.append(result)
                    
            except Exception as e:
                logger.warning(f"Prediction failed at index {i}: {e}")
                continue
        
        logger.info(f"Completed {len(self.results)} predictions")
    
    def _calculate_metrics(self):
        """Calculate backtest metrics"""
        if not self.results:
            logger.warning("No results to calculate metrics")
            return
        
        df_results = pd.DataFrame(self.results)
        
        # Basic metrics
        total_predictions = len(df_results)
        
        # Accuracy (if actual available)
        if 'correct' in df_results.columns:
            correct = df_results['correct'].dropna()
            if len(correct) > 0:
                self.accuracy = correct.sum() / len(correct)
                self.total_correct = correct.sum()
                self.total_checked = len(correct)
            else:
                self.accuracy = None
                self.total_correct = 0
                self.total_checked = 0
        else:
            self.accuracy = None
            self.total_correct = 0
            self.total_checked = 0
        
        # Signal distribution
        if 'signal' in df_results.columns:
            self.buy_signals = (df_results['signal'] == 'BUY').sum()
            self.sell_signals = (df_results['signal'] == 'SELL').sum()
        else:
            self.buy_signals = 0
            self.sell_signals = 0
        
        # Confidence statistics
        if 'confidence' in df_results.columns:
            self.avg_confidence = df_results['confidence'].mean()
            self.min_confidence = df_results['confidence'].min()
            self.max_confidence = df_results['confidence'].max()
        else:
            self.avg_confidence = 0
            self.min_confidence = 0
            self.max_confidence = 0
        
        # Model type distribution
        if 'model_type' in df_results.columns:
            self.model_types = df_results['model_type'].value_counts().to_dict()
        else:
            self.model_types = {}
        
        self.total_predictions = total_predictions
    
    def _print_results(self):
        """Print backtest results"""
        logger.info("\n" + "="*60)
        logger.info("BACKTEST RESULTS")
        logger.info("="*60)
        
        logger.info(f"\nTotal Predictions: {self.total_predictions}")
        
        if self.accuracy is not None:
            logger.info(f"Accuracy: {self.accuracy:.2%} ({self.total_correct}/{self.total_checked})")
        else:
            logger.info("Accuracy: Not available (no target data)")
        
        logger.info(f"\nSignal Distribution:")
        logger.info(f"  BUY: {self.buy_signals}")
        logger.info(f"  SELL: {self.sell_signals}")
        
        logger.info(f"\nConfidence Statistics:")
        logger.info(f"  Average: {self.avg_confidence:.2%}")
        logger.info(f"  Min: {self.min_confidence:.2%}")
        logger.info(f"  Max: {self.max_confidence:.2%}")
        
        if self.model_types:
            logger.info(f"\nModel Types Used:")
            for model_type, count in self.model_types.items():
                logger.info(f"  {model_type}: {count}")
        
        logger.info("\n" + "="*60)
    
    def save_results(self, output_file='backtest_results.csv'):
        """Save backtest results to CSV"""
        if not self.results:
            logger.warning("No results to save")
            return
        
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Test backtester
    backtester = Backtester()
    backtester.run()

