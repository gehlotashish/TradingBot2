"""
Debug script to check label distribution and target generation
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG
from models.target_generator import TargetGenerator

# Load training data
training_file = DATA_CONFIG['TRAINING_DATA_FILE']

if not os.path.exists(training_file):
    print(f"Training data not found: {training_file}")
    print("Run: python main.py --prepare-data")
    exit(1)

print("="*60)
print("LABEL DEBUGGING")
print("="*60)

df = pd.read_csv(training_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\nTotal rows: {len(df)}")
print(f"Columns: {df.columns.tolist()}")

# Check target column
if 'target' in df.columns:
    target_col = 'target'
elif 'target' in df.columns:
    target_col = 'target'
else:
    print("\n[ERROR] 'target' column not found!")
    print(f"Available columns: {df.columns.tolist()}")
    exit(1)

print(f"\n{'='*60}")
print("TARGET COLUMN ANALYSIS")
print(f"{'='*60}")

target = df[target_col]

# Check data type
print(f"\n1. Data Type: {target.dtype}")
print(f"   Unique types: {set(type(x).__name__ for x in target.unique()[:10])}")

# Check unique values
unique_vals = target.unique()
print(f"\n2. Unique Values: {sorted(unique_vals)}")
print(f"   Count of unique values: {len(unique_vals)}")

# Check for NaN
nan_count = target.isna().sum()
print(f"\n3. NaN Count: {nan_count}")

# Check distribution
print(f"\n4. Value Distribution:")
value_counts = target.value_counts()
print(value_counts)

# Check if all same
if len(unique_vals) == 1:
    print(f"\n[CRITICAL ERROR] All labels are the same: {unique_vals[0]}")
    print("   Model cannot learn - dataset has only one class!")
elif len(unique_vals) == 2:
    if set(unique_vals) == {0, 1}:
        print(f"\n[OK] Binary labels (0, 1) detected")
        print(f"   Class 0 (SELL): {value_counts.get(0, 0)}")
        print(f"   Class 1 (BUY): {value_counts.get(1, 0)}")
        
        # Check balance
        ratio = min(value_counts[0], value_counts[1]) / max(value_counts[0], value_counts[1])
        if ratio < 0.3:
            print(f"\n[WARNING] Highly imbalanced: Ratio = {ratio:.2f}")
        else:
            print(f"\n[OK] Reasonably balanced: Ratio = {ratio:.2f}")
    else:
        print(f"\n[ERROR] Binary labels but not (0, 1): {unique_vals}")
        print("   Expected: [0, 1]")
        print("   Got: {unique_vals}")
else:
    print(f"\n[ERROR] Not binary classification: {len(unique_vals)} unique values")
    print(f"   Values: {unique_vals}")

# Check sample data
print(f"\n{'='*60}")
print("SAMPLE DATA")
print(f"{'='*60}")
print("\nFirst 20 rows with target:")
sample = df[['timestamp', 'close', target_col]].head(20)
print(sample)

print("\nLast 20 rows with target:")
sample = df[['timestamp', 'close', target_col]].tail(20)
print(sample)

# Check if target makes sense
print(f"\n{'='*60}")
print("TARGET LOGIC CHECK")
print(f"{'='*60}")

# Regenerate target to see if it matches
if 'close' in df.columns:
    test_df = df[['close']].copy()
    target_gen = TargetGenerator()
    regenerated = target_gen.generate_binary_target(test_df)
    
    print(f"\nRegenerated target distribution:")
    print(regenerated.value_counts())
    
    # Compare
    if 'target' in df.columns:
        original = df['target'].dropna()
        regen_clean = regenerated.dropna()
        
        if len(original) == len(regen_clean):
            match = (original.values == regen_clean.values).sum()
            print(f"\nMatch with existing target: {match}/{len(original)} ({match/len(original)*100:.1f}%)")
        else:
            print(f"\nLength mismatch: Original={len(original)}, Regenerated={len(regen_clean)}")

print(f"\n{'='*60}")
print("RECOMMENDATIONS")
print(f"{'='*60}")

if len(unique_vals) == 1:
    print("\n[FIX] All labels are the same!")
    print("   1. Check target generation logic in models/target_generator.py")
    print("   2. Verify price data is correct")
    print("   3. Check if lookahead is too large")
elif len(unique_vals) != 2 or set(unique_vals) != {0, 1}:
    print("\n[FIX] Labels are not binary (0, 1)!")
    print("   1. Convert labels to 0 and 1")
    print("   2. Check target generation")
else:
    print("\n[OK] Labels look correct")
    print("   If model still not learning, check:")
    print("   1. Feature scaling")
    print("   2. Learning rate")
    print("   3. Model architecture")

