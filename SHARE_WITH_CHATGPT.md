# How to Share This Project with ChatGPT

## 📤 Method 1: Share This File

Simply copy and paste the contents of `PROJECT_OVERVIEW.md` to ChatGPT.

## 📤 Method 2: Share Key Code Files

Share these files in order:

1. **config.py** - Configuration
2. **main.py** - Entry point
3. **train_model.py** - Training logic
4. **feature_engineering/feature_generator.py** - Features
5. **models/target_generator.py** - Targets
6. **models/deep_learning/multi_head_lstm.py** - Multi-head architecture
7. **live_engine/prediction_engine.py** - Prediction engine

## 📤 Method 3: Create a Summary Script

Run this to generate a complete project summary:

```python
# generate_project_summary.py
import os
import glob

def get_file_structure():
    """Get project file structure"""
    structure = {}
    
    # Key directories
    dirs = ['data_pipeline', 'feature_engineering', 'models', 
            'models/deep_learning', 'live_engine', 'telegram_bot']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            files = glob.glob(f"{dir_name}/*.py")
            structure[dir_name] = [os.path.basename(f) for f in files]
    
    return structure

def generate_summary():
    """Generate project summary"""
    summary = "# AI Trading Signal System - Project Summary\n\n"
    
    summary += "## Project Structure\n\n"
    structure = get_file_structure()
    for dir_name, files in structure.items():
        summary += f"### {dir_name}/\n"
        for file in files:
            summary += f"- {file}\n"
        summary += "\n"
    
    summary += "## Key Features\n\n"
    summary += "1. Multi-head LSTM model (4 outputs: trend, direction, magnitude, volatility)\n"
    summary += "2. 15-minute prediction (more predictable than 1-min)\n"
    summary += "3. 27 technical features\n"
    summary += "4. Telegram integration\n"
    summary += "5. SmartAPI data fetching\n\n"
    
    summary += "## Current Status\n\n"
    summary += "- RandomForest: 70.6% accuracy\n"
    summary += "- LSTM: ~50% accuracy (learning issue)\n"
    summary += "- Multi-head: Architecture ready\n\n"
    
    return summary

if __name__ == "__main__":
    summary = generate_summary()
    print(summary)
    with open("PROJECT_SUMMARY.txt", "w", encoding="utf-8") as f:
        f.write(summary)
    print("\nSummary saved to PROJECT_SUMMARY.txt")
```

## 📋 Quick Share Template

Copy this and paste to ChatGPT:

```
I have an AI Trading Signal System project. Here's the overview:

PROJECT: AI Trading Signal System (BankNifty + Nifty50)

FEATURES:
- Fetches data from Angel One SmartAPI
- Generates 27 technical features (EMA, RSI, MACD, ATR, VWAP, etc.)
- Trains multiple models: RandomForest, XGBoost, LSTM, Multi-Head LSTM
- Multi-head LSTM predicts 4 things: trend continuation, direction, magnitude, volatility
- Sends signals to Telegram

ARCHITECTURE:
- Data Pipeline → Feature Engineering → Model Training → Live Prediction → Telegram

CURRENT STATUS:
- RandomForest: 70.6% accuracy ✅
- LSTM: ~50% accuracy (not learning) ⚠️
- Multi-head LSTM: Architecture ready, needs retraining

ISSUES:
1. LSTM stuck at 50% accuracy (random guessing)
2. Multi-head indexing bug (just fixed)
3. Feature generator NaN bug (just fixed)

FILES:
- config.py: Configuration
- train_model.py: Training
- feature_engineering/feature_generator.py: Features
- models/deep_learning/multi_head_lstm.py: Multi-head model
- live_engine/prediction_engine.py: Predictions

I need help with:
[Your specific question]
```

## 🔗 Best Approach

**Recommended**: Share `PROJECT_OVERVIEW.md` file content + specific code files you want help with.

For example:
1. "Here's my project overview: [paste PROJECT_OVERVIEW.md]"
2. "I need help with this file: [paste specific file]"
3. "The issue is: [describe problem]"

This gives ChatGPT complete context!

