"""
Main Entry Point for AI Trading Signal System
"""

import os
import sys
import argparse
from logzero import logger

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import validate_config
from train_model import prepare_training_data, train_model
from live_engine.live_trading_engine import LiveTradingEngine


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='AI Trading Signal System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare training data
  python main.py --prepare-data
  
  # Train model
  python main.py --train
  
  # Prepare data and train
  python main.py --full
  
  # Run live engine
  python main.py --live
  
  # Test Telegram
  python main.py --test-telegram
        """
    )
    
    parser.add_argument('--prepare-data', action='store_true',
                       help='Prepare training data from historical data')
    parser.add_argument('--train', action='store_true',
                       help='Train ML model')
    parser.add_argument('--full', action='store_true',
                       help='Prepare data and train model')
    parser.add_argument('--live', action='store_true',
                       help='Run live trading engine')
    parser.add_argument('--test-telegram', action='store_true',
                       help='Test Telegram bot configuration')
    parser.add_argument('--model-type', choices=['random_forest', 'xgboost'],
                       default='random_forest', help='Model type for training')
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure all required variables are set")
        return 1
    
    # Execute requested action
    if args.prepare_data:
        logger.info("Preparing training data...")
        prepare_training_data()
    
    elif args.train:
        logger.info("Training model...")
        train_model(model_type=args.model_type)
    
    elif args.full:
        logger.info("Preparing data and training model...")
        df = prepare_training_data()
        if df is not None:
            train_model(df, model_type=args.model_type)
    
    elif args.live:
        logger.info("Starting live trading engine...")
        engine = LiveTradingEngine()
        engine.start()
    
    elif args.test_telegram:
        logger.info("Testing Telegram configuration...")
        from telegram_bot.telegram_sender import TelegramSender
        sender = TelegramSender()
        sender.send_test_message()
    
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

