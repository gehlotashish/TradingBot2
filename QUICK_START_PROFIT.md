# Quick Start - Profit-Focused Configuration

## 🎯 Recommended Settings for Maximum Profit

### Step 1: Update Config (`config.py`)

**Best Configuration for Profit**:
```python
TARGET_CONFIG = {
    'TARGET_TYPE': 'multi_timeframe',  # 15-minute prediction
    'LOOKAHEAD_LONG': 15,  # Predict 15 minutes ahead
    'LOOKAHEAD_SHORT': 5,  # Option for 5-min
    'TREND_WINDOW': 5,
}
```

### Step 2: Reprepare Data
```bash
python main.py --prepare-data
```

This will now:
- Generate 15-minute targets (instead of 1-min)
- Add VWAP, Volume Delta, Trend features
- Create more predictable dataset

### Step 3: Retrain Model
```bash
python main.py --train --train-lstm
```

## 📊 Expected Results

### Before (1-min prediction):
- Accuracy: ~50% (random)
- Profit: Unpredictable

### After (15-min prediction):
- Accuracy: 55-62% (achievable)
- Profit: More consistent

## 🔄 Alternative Configurations

### Option A: Trend Continuation (Most Predictable)
```python
TARGET_CONFIG = {
    'TARGET_TYPE': 'trend',
    'LOOKAHEAD_LONG': 15,
    'TREND_WINDOW': 5,
}
```
**Expected**: 58-65% accuracy

### Option B: Regression (Magnitude)
```python
TARGET_CONFIG = {
    'TARGET_TYPE': 'regression',
    'LOOKAHEAD_LONG': 15,
}
```
**Expected**: Better profit calculation

### Option C: 5-Minute (Balance)
```python
TARGET_CONFIG = {
    'TARGET_TYPE': 'multi_timeframe',
    'LOOKAHEAD_LONG': 5,  # 5 minutes
}
```
**Expected**: 53-58% accuracy

## ✅ New Features Added

1. **VWAP** - Volume Weighted Average Price
2. **Volume Delta** - Change in volume
3. **Volume Momentum** - Rate of volume change
4. **Trend Direction** - Current trend (up/down)
5. **Trend Strength** - How strong the trend is
6. **EMA Crossovers** - Bullish/bearish signals

## 🚀 Ready to Use!

All improvements are implemented. Just:
1. Update config (if needed)
2. Reprepare data
3. Retrain model
4. Run live engine

---

**Status**: ✅ Profit-Focused Improvements Complete!

