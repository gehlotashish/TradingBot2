"""
Step-by-step project startup guide
"""

import os
import sys

def print_step(step_num, title):
    print("\n" + "="*60)
    print(f"STEP {step_num}: {title}")
    print("="*60)

def check_dependencies():
    """Step 1: Check if dependencies are installed"""
    print_step(1, "CHECKING DEPENDENCIES")
    
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'dotenv': 'python-dotenv',
        'SmartApi': 'SmartApi',
        'pyotp': 'pyotp',
        'requests': 'requests',
        'logzero': 'logzero',
        'schedule': 'schedule',
    }
    
    missing = []
    for module, package in required_packages.items():
        try:
            __import__(module)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing.append(package)
    
    if missing:
        print(f"\n[ACTION REQUIRED] Install missing packages:")
        print(f"pip install {' '.join(missing)}")
        return False
    else:
        print("\n[OK] All dependencies installed!")
        return True

def check_config():
    """Step 2: Check configuration"""
    print_step(2, "CHECKING CONFIGURATION")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    required_vars = {
        'HISTORICAL_DATA_API_KEY': 'SmartAPI Historical Key',
        'LIVE_DATA_API_KEY': 'SmartAPI Live Key',
        'CLIENT_CODE': 'Client Code',
        'PIN': 'PIN',
        'TOTP_SECRET': 'TOTP Secret',
        'TELEGRAM_BOT_TOKEN': 'Telegram Bot Token',
        'TELEGRAM_CHAT_ID': 'Telegram Chat ID',
    }
    
    missing = []
    for var, name in required_vars.items():
        value = os.getenv(var, '')
        if not value or value.startswith('your_'):
            print(f"[MISSING] {name}")
            missing.append(var)
        else:
            # Show partial value for security
            if 'TOKEN' in var or 'SECRET' in var or 'KEY' in var:
                display = value[:10] + "..." if len(value) > 10 else "***"
            else:
                display = value
            print(f"[OK] {name}: {display}")
    
    if missing:
        print(f"\n[ACTION REQUIRED] Configure missing variables in .env file")
        return False
    else:
        print("\n[OK] Configuration complete!")
        return True

def check_training_data():
    """Step 3: Check if training data exists"""
    print_step(3, "CHECKING TRAINING DATA")
    
    from config import DATA_CONFIG
    training_file = DATA_CONFIG['TRAINING_DATA_FILE']
    
    if os.path.exists(training_file):
        import pandas as pd
        df = pd.read_csv(training_file)
        print(f"[OK] Training data found: {training_file}")
        print(f"     Rows: {len(df):,}")
        print(f"     Columns: {len(df.columns)}")
        if 'target' in df.columns:
            print(f"     Target distribution:")
            print(f"       BUY (1): {(df['target'] == 1).sum():,}")
            print(f"       SELL (0): {(df['target'] == 0).sum():,}")
        return True
    else:
        print(f"[MISSING] Training data not found: {training_file}")
        print(f"         Run: python main.py --prepare-data")
        return False

def check_model():
    """Step 4: Check if model exists"""
    print_step(4, "CHECKING TRAINED MODEL")
    
    from config import MODEL_CONFIG
    model_file = MODEL_CONFIG['MODEL_FILE']
    
    if os.path.exists(model_file):
        file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
        print(f"[OK] Model found: {model_file}")
        print(f"     Size: {file_size:.2f} MB")
        return True
    else:
        print(f"[MISSING] Model not found: {model_file}")
        print(f"         Run: python main.py --train")
        return False

def main():
    print("\n" + "="*60)
    print("AI TRADING SIGNAL SYSTEM - STARTUP CHECK")
    print("="*60)
    
    # Step 1: Dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n[STOP] Please install missing dependencies first")
        return
    
    # Step 2: Configuration
    config_ok = check_config()
    if not config_ok:
        print("\n[STOP] Please configure .env file first")
        return
    
    # Step 3: Training Data
    data_ok = check_training_data()
    
    # Step 4: Model
    model_ok = check_model()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Dependencies: {'[OK]' if deps_ok else '[MISSING]'}")
    print(f"Configuration: {'[OK]' if config_ok else '[MISSING]'}")
    print(f"Training Data: {'[OK]' if data_ok else '[MISSING]'}")
    print(f"Trained Model: {'[OK]' if model_ok else '[MISSING]'}")
    print("="*60)
    
    if deps_ok and config_ok:
        if not data_ok:
            print("\n[NEXT STEP] Prepare training data:")
            print("   python main.py --prepare-data")
        elif not model_ok:
            print("\n[NEXT STEP] Train the model:")
            print("   python main.py --train")
        else:
            print("\n[READY] System is ready to run!")
            print("   Start live engine:")
            print("   python main.py --live")

if __name__ == "__main__":
    main()

