"""
Model Training Script
Fetches historical data, generates features, creates target, and trains model
"""

import os
import sys
import pandas as pd
from logzero import logger
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import DATA_CONFIG, TRADING_SYMBOLS, validate_config
from data_pipeline.data_fetcher import DataFetcher
from data_pipeline.data_cleaner import DataCleaner
from feature_engineering.feature_generator import FeatureGenerator
from models.target_generator import TargetGenerator
from models.model_trainer import ModelTrainer


def prepare_training_data():
    """
    Complete pipeline to prepare training data:
    1. Fetch historical data for all symbols
    2. Clean data
    3. Generate features
    4. Generate target
    5. Save to CSV
    """
    logger.info("="*60)
    logger.info("PREPARING TRAINING DATA")
    logger.info("="*60)
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return None
    
    # Initialize components
    fetcher = DataFetcher()
    cleaner = DataCleaner()
    generator = FeatureGenerator()
    target_gen = TargetGenerator()
    
    # Connect to API
    if not fetcher.connect():
        logger.error("Failed to connect to SmartAPI")
        return None
    
    all_data = []
    
    # Process each symbol
    for symbol_key, symbol_info in TRADING_SYMBOLS.items():
        logger.info(f"\nProcessing {symbol_key} ({symbol_info['name']})...")
        
        # Fetch historical data
        df = fetcher.fetch_training_data(symbol_key)
        if df is None or df.empty:
            logger.warning(f"No data fetched for {symbol_key}")
            continue
        
        logger.info(f"Fetched {len(df)} candles")
        
        # Clean data
        df = cleaner.clean_ohlcv(df)
        if df.empty:
            logger.warning(f"No data after cleaning for {symbol_key}")
            continue
        
        logger.info(f"Cleaned data: {len(df)} candles")
        
        # Generate features
        df = generator.generate_features(df)
        if df.empty:
            logger.warning(f"No data after feature generation for {symbol_key}")
            continue
        
        logger.info(f"Generated features: {len(df)} candles")
        
        # Generate target
        df['target'] = target_gen.generate_binary_target(df)
        
        # Add symbol identifier
        df['symbol'] = symbol_key
        
        # Remove rows with NaN target
        df = df.dropna(subset=['target']).reset_index(drop=True)
        
        logger.info(f"Final data: {len(df)} candles with target")
        
        all_data.append(df)
    
    # Combine all symbols
    if not all_data:
        logger.error("No data prepared for any symbol")
        return None
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Final cleanup
    combined_df = combined_df.dropna().reset_index(drop=True)
    
    logger.info(f"\nTotal training data: {len(combined_df)} samples")
    logger.info(f"Features: {len(generator.get_feature_columns())}")
    logger.info(f"Target distribution:")
    logger.info(combined_df['target'].value_counts())
    
    # Save to CSV
    os.makedirs(os.path.dirname(DATA_CONFIG['TRAINING_DATA_FILE']), exist_ok=True)
    combined_df.to_csv(DATA_CONFIG['TRAINING_DATA_FILE'], index=False)
    logger.info(f"\nTraining data saved to: {DATA_CONFIG['TRAINING_DATA_FILE']}")
    
    return combined_df


def train_model(df=None, model_type='random_forest'):
    """
    Train ML model on prepared data
    
    Args:
        df: DataFrame with training data (if None, load from file)
        model_type: 'random_forest' or 'xgboost'
    """
    logger.info("="*60)
    logger.info("TRAINING ML MODEL")
    logger.info("="*60)
    
    # Load data if not provided
    if df is None:
        training_file = DATA_CONFIG['TRAINING_DATA_FILE']
        if not os.path.exists(training_file):
            logger.error(f"Training data file not found: {training_file}")
            logger.info("Run prepare_training_data() first")
            return None
        
        logger.info(f"Loading training data from {training_file}")
        df = pd.read_csv(training_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Train model
    trainer = ModelTrainer()
    metrics = trainer.train_and_save(df, model_type=model_type)
    
    logger.info("\n" + "="*60)
    logger.info("MODEL TRAINING COMPLETED")
    logger.info("="*60)
    logger.info(f"Model Type: {model_type}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    from config import MODEL_CONFIG
    logger.info(f"Model saved to: {MODEL_CONFIG['MODEL_FILE']}")
    
    return metrics


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AI Trading Signal Model')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare training data')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--model-type', choices=['random_forest', 'xgboost'], default='random_forest',
                       help='Model type to train')
    parser.add_argument('--full', action='store_true', help='Prepare data and train model')
    
    args = parser.parse_args()
    
    if args.full or args.prepare_data:
        df = prepare_training_data()
        if df is not None and (args.full or args.train):
            train_model(df, model_type=args.model_type)
    elif args.train:
        train_model(model_type=args.model_type)
    else:
        # Default: do both
        df = prepare_training_data()
        if df is not None:
            train_model(df, model_type=args.model_type)

