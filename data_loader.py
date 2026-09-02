"""
Data loading and splitting module.

Pipeline:
1. Split FIRST, then resample only the training set
2. Use the SAME test set across ALL models (ML, DL, BERT)
3. Proper validation set (separate from test set)
4. Near-duplicate detection before splitting
5. Stratified splitting to preserve class distribution
"""

import os
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from collections import Counter

import config
from preprocess import clean_text


def load_and_clean(dataset_name: str) -> pd.DataFrame:
    """
    Load a dataset CSV and perform basic cleaning.
    Returns a DataFrame with columns ['text', 'label'] (standardized).
    """
    path = config.DATASET_PATHS[dataset_name]
    cols = config.DATASET_COLUMNS[dataset_name]

    if not os.path.exists(path) or os.path.getsize(path) < 100:
        raise ValueError(f'Dataset at {path} is missing or empty (size < 100 bytes). Please download the actual dataset from Kaggle.')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            f"Please download from Kaggle — see dataset/README.md"
        )

    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')

    # Drop unnamed index columns
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Standardize column names
    df = df.rename(columns={cols['text']: 'text', cols['label']: 'label'})

    # Drop rows with missing text or label
    df.dropna(subset=['text', 'label'], inplace=True)

    # Drop exact duplicates
    df.drop_duplicates(subset=['text'], inplace=True)

    df.reset_index(drop=True, inplace=True)

    print(f"[{dataset_name}] Loaded {len(df)} samples after cleaning")
    print(f"  Class distribution:\n{df['label'].value_counts().to_string()}\n")

    return df[['text', 'label']]


def check_near_duplicates(df: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Detect and remove near-duplicate texts using MinHash-like fingerprinting.
    Uses character n-gram hashing for fast approximate duplicate detection.
    
    Returns cleaned DataFrame with near-duplicates removed.
    """
    print("  Checking for near-duplicates...")

    # Create fingerprints using character trigrams
    def text_fingerprint(text: str) -> set:
        text = text.lower().strip()
        if len(text) < 3:
            return {text}
        return set(text[i:i+3] for i in range(len(text) - 2))

    fingerprints = df['text'].apply(text_fingerprint)
    
    # Compare using Jaccard similarity (optimized: only check within same label)
    to_remove = set()
    for label in df['label'].unique():
        label_indices = df[df['label'] == label].index.tolist()
        for i in range(len(label_indices)):
            if label_indices[i] in to_remove:
                continue
            for j in range(i + 1, min(i + 50, len(label_indices))):  # Check nearby rows
                idx_i, idx_j = label_indices[i], label_indices[j]
                if idx_j in to_remove:
                    continue
                fp_i, fp_j = fingerprints[idx_i], fingerprints[idx_j]
                if not fp_i or not fp_j:
                    continue
                jaccard = len(fp_i & fp_j) / len(fp_i | fp_j)
                if jaccard >= threshold:
                    to_remove.add(idx_j)

    if to_remove:
        print(f"  Removed {len(to_remove)} near-duplicate texts")
        df = df.drop(index=to_remove).reset_index(drop=True)
    else:
        print("  No near-duplicates found")

    return df


def split_data(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    THE CORRECT SPLITTING PIPELINE:
    
    1. Clean and deduplicate (already done in load_and_clean)
    2. Optionally check for near-duplicates
    3. SPLIT FIRST into train/val/test using stratification
    4. THEN resample ONLY the training set to balance classes
    5. Apply text preprocessing AFTER splitting
    
    Returns dict with keys: 
        'X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test',
        'X_train_clean', 'X_val_clean', 'X_test_clean',
        'classes', 'num_classes'
    """
    print(f"[{dataset_name}] Splitting data (CORRECT pipeline)...")

    # ── Step 1: Near-duplicate check ──
    df = check_near_duplicates(df, config.DUPLICATE_SIMILARITY_THRESHOLD)

    X = df['text']
    y = df['label']

    # ── Step 2: SPLIT FIRST (stratified) ──
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    # Second split: train vs val (from trainval)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y_trainval
    )

    print(f"  Before resampling:")
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"    Train class dist: {dict(Counter(y_train))}")

    # ── Step 3: RESAMPLE ONLY THE TRAINING SET ──
    train_df = pd.DataFrame({'text': X_train.values, 'label': y_train.values})
    max_count = train_df['label'].value_counts().max()

    resampled_parts = []
    for label in train_df['label'].unique():
        subset = train_df[train_df['label'] == label]
        if len(subset) < max_count:
            subset_resampled = resample(
                subset, replace=True, n_samples=max_count,
                random_state=config.RANDOM_STATE
            )
            resampled_parts.append(subset_resampled)
        else:
            resampled_parts.append(subset)

    train_df = pd.concat(resampled_parts).sample(
        frac=1, random_state=config.RANDOM_STATE
    ).reset_index(drop=True)

    X_train = train_df['text']
    y_train = train_df['label']

    print(f"  After resampling (TRAINING SET ONLY):")
    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"    Train class dist: {dict(Counter(y_train))}")

    # ── Step 4: Text preprocessing ──
    print("  Applying text preprocessing...")
    X_train_clean = X_train.apply(clean_text)
    X_val_clean = X_val.apply(clean_text)
    X_test_clean = X_test.apply(clean_text)

    classes = sorted(y.unique())
    num_classes = len(classes)

    # ── Verify no leakage ──
    train_texts = set(X_train.values)
    test_texts = set(X_test.values)
    val_texts = set(X_val.values)

    train_test_overlap = train_texts & test_texts
    train_val_overlap = train_texts & val_texts

    if train_test_overlap:
        # This can happen from resampling — but only resampled copies, not originals
        # Check if overlaps are from the original (pre-resample) train set
        print(f"  WARNING: {len(train_test_overlap)} texts appear in both train and test!")
    else:
        print("  VERIFIED: No text overlap between train and test sets")

    if train_val_overlap:
        print(f"  WARNING: {len(train_val_overlap)} texts appear in both train and val!")
    else:
        print("  VERIFIED: No text overlap between train and val sets")

    result = {
        'X_train': X_train.reset_index(drop=True),
        'X_val': X_val.reset_index(drop=True),
        'X_test': X_test.reset_index(drop=True),
        'y_train': y_train.reset_index(drop=True),
        'y_val': y_val.reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
        'X_train_clean': X_train_clean.reset_index(drop=True),
        'X_val_clean': X_val_clean.reset_index(drop=True),
        'X_test_clean': X_test_clean.reset_index(drop=True),
        'classes': classes,
        'num_classes': num_classes,
    }

    print(f"  Final sizes — Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  Classes ({num_classes}): {classes}\n")

    return result


def prepare_dataset(dataset_name: str) -> dict:
    """
    Full pipeline: load -> clean -> split -> resample (train only).
    Single entry point for all models.
    """
    df = load_and_clean(dataset_name)
    return split_data(df, dataset_name)
