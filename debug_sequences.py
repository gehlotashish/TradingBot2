"""
Debug sequence creation and target alignment
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import DATA_CONFIG, DL_CONFIG
from models.deep_learning.sequence_preprocessor import SequencePreprocessor

# Load training data
training_file = DATA_CONFIG['TRAINING_DATA_FILE']

if not os.path.exists(training_file):
    print(f"Training data not found: {training_file}")
    exit(1)

print("="*60)
print("SEQUENCE CREATION DEBUG")
print("="*60)

df = pd.read_csv(training_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"\nTotal rows: {len(df)}")
print(f"Target distribution in original data:")
print(df['target'].value_counts())

# Create sequences
sequence_length = DL_CONFIG['SEQUENCE_LENGTH']
preprocessor = SequencePreprocessor(sequence_length=sequence_length)

print(f"\n{'='*60}")
print(f"CREATING SEQUENCES (length={sequence_length})")
print(f"{'='*60}")

X, y = preprocessor.create_sequences(df)

if X is None or y is None:
    print("[ERROR] Failed to create sequences")
    exit(1)

print(f"\nSequences created: {len(X)}")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"y dtype: {y.dtype}")

# Check target distribution in sequences
print(f"\n{'='*60}")
print("TARGET DISTRIBUTION IN SEQUENCES")
print(f"{'='*60}")

unique_y, counts_y = np.unique(y, return_counts=True)
print(f"Unique values in y: {unique_y}")
print(f"Counts: {dict(zip(unique_y, counts_y))}")

if len(unique_y) == 1:
    print(f"\n[CRITICAL ERROR] All sequence targets are the same: {unique_y[0]}")
    print("   This is why model cannot learn!")
elif len(unique_y) == 2:
    ratio = min(counts_y) / max(counts_y)
    print(f"\nTarget balance ratio: {ratio:.3f}")
    if ratio < 0.3:
        print("[WARNING] Highly imbalanced sequences")
    else:
        print("[OK] Sequences are reasonably balanced")

# Verify target alignment
print(f"\n{'='*60}")
print("TARGET ALIGNMENT CHECK")
print(f"{'='*60}")

# Check first few sequences
print("\nFirst 5 sequences:")
for i in range(min(5, len(X))):
    seq_start_idx = i
    target_idx = i + sequence_length
    
    if target_idx < len(df):
        original_target = df['target'].iloc[target_idx]
        sequence_target = y[i]
        
        print(f"Sequence {i}:")
        print(f"  Sequence covers rows: {seq_start_idx} to {seq_start_idx + sequence_length - 1}")
        print(f"  Target row: {target_idx}")
        print(f"  Original target at row {target_idx}: {original_target}")
        print(f"  Sequence target y[{i}]: {sequence_target}")
        print(f"  Match: {original_target == sequence_target}")
        print()

# Check if targets are correctly aligned
print("\nVerifying target alignment...")
matches = 0
total = min(100, len(y))  # Check first 100

for i in range(total):
    target_idx = i + sequence_length
    if target_idx < len(df):
        if df['target'].iloc[target_idx] == y[i]:
            matches += 1

print(f"Target alignment: {matches}/{total} matches ({matches/total*100:.1f}%)")

if matches < total * 0.9:
    print("[ERROR] Target alignment is wrong!")
    print("   Targets in sequences don't match original data")
else:
    print("[OK] Target alignment looks correct")

# Check feature values
print(f"\n{'='*60}")
print("FEATURE VALUE CHECK")
print(f"{'='*60}")

print(f"\nX statistics:")
print(f"  Min: {X.min():.4f}")
print(f"  Max: {X.max():.4f}")
print(f"  Mean: {X.mean():.4f}")
print(f"  Std: {X.std():.4f}")
print(f"  NaN count: {np.isnan(X).sum()}")
print(f"  Inf count: {np.isinf(X).sum()}")

if np.isnan(X).any() or np.isinf(X).any():
    print("[ERROR] NaN or Inf values in features!")
elif X.max() > 1000 or X.min() < -1000:
    print("[WARNING] Feature values seem too large/small")
    print("   Consider better normalization")
else:
    print("[OK] Feature values look reasonable")

print(f"\n{'='*60}")
print("DIAGNOSIS")
print(f"{'='*60}")

if len(unique_y) == 1:
    print("\n[ROOT CAUSE] All sequence targets are the same!")
    print("   Fix: Check sequence creation logic")
elif np.isnan(X).any() or np.isinf(X).any():
    print("\n[ROOT CAUSE] NaN/Inf in features!")
    print("   Fix: Better data cleaning/normalization")
elif matches < total * 0.9:
    print("\n[ROOT CAUSE] Target misalignment!")
    print("   Fix: Correct target indexing in sequence creation")
else:
    print("\n[OK] Sequences look correct")
    print("   If model still not learning, try:")
    print("   1. Different model architecture")
    print("   2. Different learning rate")
    print("   3. More training data")
    print("   4. Feature engineering improvements")

