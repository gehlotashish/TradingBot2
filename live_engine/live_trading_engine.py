"""
Live Trading Engine
Main engine that runs predictions and sends signals
"""

import time
import os
import sys
from datetime import datetime
from logzero import logger
import schedule

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRADING_SYMBOLS, SCHEDULER_CONFIG, SIGNAL_CONFIG
from live_engine.prediction_engine import PredictionEngine
from telegram_bot.telegram_sender import TelegramSender


class LiveTradingEngine:
    """Main live trading engine"""
    
    def __init__(self):
        self.prediction_engine = None
        self.telegram_sender = None
        self.running = False
        self.last_predictions = {}  # Store last predictions per symbol
    
    def initialize(self):
        """Initialize engine components"""
        logger.info("Initializing Live Trading Engine...")
        
        try:
            self.prediction_engine = PredictionEngine()
            logger.info("✓ Prediction engine initialized")
        except Exception as e:
            logger.error(f"Failed to initialize prediction engine: {e}")
            return False
        
        try:
            self.telegram_sender = TelegramSender()
            logger.info("✓ Telegram sender initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram sender: {e}")
            return False
        
        logger.info("Live Trading Engine initialized successfully")
        return True
    
    def process_symbol(self, symbol_key):
        """
        Process a single symbol: predict and send signal if needed
        
        Args:
            symbol_key: Key from TRADING_SYMBOLS
        """
        try:
            logger.info(f"Processing {symbol_key}...")
            
            # Generate signal
            signal_data = self.prediction_engine.generate_signal(symbol_key)
            
            if signal_data is None:
                logger.warning(f"No signal generated for {symbol_key}")
                return
            
            # Store prediction
            self.last_predictions[symbol_key] = signal_data
            
            # Check if signal should be sent
            confidence = signal_data.get('confidence', 0)
            if confidence >= SIGNAL_CONFIG['MIN_CONFIDENCE']:
                # Send signal
                success = self.telegram_sender.send_signal(signal_data)
                if success:
                    logger.info(f"Signal sent for {symbol_key}: {signal_data['signal']} ({confidence:.2%})")
                else:
                    logger.warning(f"Failed to send signal for {symbol_key}")
            else:
                logger.info(f"Signal confidence {confidence:.2%} below threshold for {symbol_key}")
            
        except Exception as e:
            logger.exception(f"Error processing {symbol_key}: {e}")
    
    def run_prediction_cycle(self):
        """Run prediction cycle for all symbols"""
        logger.info(f"\n{'='*60}")
        logger.info(f"PREDICTION CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*60}")
        
        for symbol_key in TRADING_SYMBOLS.keys():
            self.process_symbol(symbol_key)
            time.sleep(1)  # Small delay between symbols
    
    def start(self):
        """Start the live trading engine"""
        if not self.initialize():
            logger.error("Failed to initialize engine")
            return False
        
        logger.info("="*60)
        logger.info("LIVE TRADING ENGINE STARTED")
        logger.info("="*60)
        logger.info(f"Prediction interval: {SCHEDULER_CONFIG['PREDICTION_INTERVAL']} seconds")
        logger.info(f"Symbols: {', '.join(TRADING_SYMBOLS.keys())}")
        logger.info("="*60)
        
        self.running = True
        
        # Schedule predictions
        interval = SCHEDULER_CONFIG['PREDICTION_INTERVAL']
        schedule.every(interval).seconds.do(self.run_prediction_cycle)
        
        # Run initial prediction
        self.run_prediction_cycle()
        
        # Main loop
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("\nStopping Live Trading Engine...")
            self.stop()
    
    def stop(self):
        """Stop the live trading engine"""
        self.running = False
        logger.info("Live Trading Engine stopped")
    
    def get_status(self):
        """Get current engine status"""
        return {
            'running': self.running,
            'last_predictions': self.last_predictions,
            'symbols': list(TRADING_SYMBOLS.keys())
        }


if __name__ == "__main__":
    # Run live trading engine
    engine = LiveTradingEngine()
    
    try:
        engine.start()
    except Exception as e:
        logger.exception(f"Error running engine: {e}")

