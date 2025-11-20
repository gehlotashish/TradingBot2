"""
Test prediction engine with a single prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from live_engine.prediction_engine import PredictionEngine
from logzero import logger

print("="*60)
print("TESTING PREDICTION ENGINE")
print("="*60)
print()

try:
    print("Loading prediction engine...")
    engine = PredictionEngine()
    print("[OK] Engine loaded")
    print()
    
    # Test prediction for BANKNIFTY
    print("Generating prediction for BANKNIFTY...")
    result = engine.generate_signal('BANKNIFTY')
    
    if result:
        print()
        print("="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Symbol: {result['symbol']}")
        print(f"Signal: {result['signal']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Current Price: ₹{result['current_price']:.2f}")
        print(f"Trigger Price: ₹{result['trigger_price']:.2f}")
        print(f"Reason: {result['reason']}")
        print()
        print("Indicators:")
        for key, value in result['indicators'].items():
            if value is not None:
                print(f"  {key}: {value:.2f}")
        print("="*60)
        print()
        print("[OK] Prediction engine is working!")
    else:
        print("[ERROR] Failed to generate prediction")
        
except Exception as e:
    logger.exception(f"Error testing prediction: {e}")
    print(f"[ERROR] {e}")

