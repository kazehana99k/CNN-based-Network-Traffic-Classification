"""
Data preprocessing module: read Moore dataset, parse ARFF files, fill missing values, balance class distribution.
Corresponds to paper Section 3.1 (Data Preparation and Preprocessing) and Section 4.2 (Dataset Preprocessing).
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, MOORE_FILES, LIST_Y,
    ARFF_HEADER_LINES, ORIGINAL_FEATURE_NUM, PADDING_NUM, TOTAL_FEATURES,
    MAX_QUESTION_MARKS, TEST_SIZE, RANDOM_STATE,
    BALANCED_COUNTS, NOISE_AUGMENT_CLASSES, NOISE_MEAN, NOISE_STD,
    NUM_CLASSES,
)


def parse_arff_line(line):
    """Parse one line in an ARFF file, returns (feature list, label string) or None.

    Strategy:
    1. Replace 'Y'/'N' with '1'/'0'
    2. Skip the line if it contains more than MAX_QUESTION_MARKS missing values
    3. Compute the mean of all known features and use it to fill in '?'
    4. Pad zeros if needed to align with TOTAL_FEATURES
    """
    line = line.strip()
    if not line or line.startswith('%') or line.startswith('@'):
        return None

    parts = line.split(',')

    # Only do Y/N -> 1/0 on feature fields (do not corrupt the class label, which contains an 'N')
    feat_str_list = [p.replace('Y', '1').replace('N', '0') for p in parts[:-1]]
    label = parts[-1].strip()

    if label == '?':
        return None
    if len([p for p in feat_str_list if p == '?']) > MAX_QUESTION_MARKS:
        return None

    known_values = []
    for v in feat_str_list:
        if v != '?':
            try:
                known_values.append(float(v))
            except ValueError:
                return None

    if not known_values:
        return None

    mean_val = sum(known_values) / len(known_values)

    features = []
    for v in feat_str_list:
        if v == '?':
            features.append(mean_val)
        else:
            try:
                features.append(float(v))
            except ValueError:
                return None

    # Truncate or pad to fixed feature size
    if len(features) > ORIGINAL_FEATURE_NUM:
        features = features[:ORIGINAL_FEATURE_NUM]
    elif len(features) < ORIGINAL_FEATURE_NUM:
        features += [0.0] * (ORIGINAL_FEATURE_NUM - len(features))

    features += [0.0] * PADDING_NUM

    return features, label


def load_moore_files(filenames=None, data_dir=None):
    """Load multiple Moore dataset files, return X (N, 256) and y (N,).

    Args:
        filenames: list of arff file names; defaults to config.MOORE_FILES
        data_dir: data directory path; defaults to config.DATA_DIR
    """
    filenames = filenames or MOORE_FILES
    data_dir = data_dir or DATA_DIR

    X, Y = [], []
    for fname in filenames:
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f'[Warning] File not found: {path}, skipping.')
            continue

        print(f'[Info] Loading {fname} ...')
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        # Skip ARFF header
        data_lines = lines[ARFF_HEADER_LINES:]

        cnt = 0
        for line in data_lines:
            result = parse_arff_line(line)
            if result is None:
                continue
            features, label = result

            # Normalize label name (note 4.2 in the paper)
            label = label.replace('FTP-CONTROL', 'FTP-CONTROL_')
            label = label.replace('INTERACTIVE', 'INTERACTIVE_')
            label = label.replace('FTP-CONTROL__', 'FTP-CONTROL_')  # prevent double underscore
            label = label.replace('INTERACTIVE__', 'INTERACTIVE_')

            if label not in LIST_Y:
                continue

            X.append(features)
            Y.append(LIST_Y.index(label))
            cnt += 1

        print(f'[Info]   {fname} parsed {cnt} samples.')

    if not X:
        raise FileNotFoundError(
            f'Failed to load any Moore samples. Make sure the dataset files are under {data_dir}.\n'
            'See README.md for the data download instructions.'
        )

    return np.array(X, dtype=np.float64), np.array(Y, dtype=np.int32)


def balance_dataset(X, y, target_counts=None):
    """Balance the data based on Section 3.1.2 Table 3.2 of the paper.

    - Down-sample heavily over-represented classes such as WWW
    - Use mean-filling (add Gaussian white noise) to augment under-represented classes
    """
    target_counts = target_counts or BALANCED_COUNTS
    rng = np.random.default_rng(RANDOM_STATE)

    new_X, new_y = [], []
    unique_labels = np.unique(y)

    for lbl_idx in unique_labels:
        lbl_name = LIST_Y[lbl_idx]
        mask = (y == lbl_idx)
        X_cls = X[mask]
        target = target_counts.get(lbl_name, len(X_cls))

        if len(X_cls) >= target:
            idx = rng.choice(len(X_cls), size=target, replace=False)
            X_sel = X_cls[idx]
        else:
            need = target - len(X_cls)
            if lbl_name in NOISE_AUGMENT_CLASSES and len(X_cls) > 0:
                # Mean-filling algorithm: copy existing samples then add Gaussian noise
                base_idx = rng.choice(len(X_cls), size=need, replace=True)
                base = X_cls[base_idx]
                noise = rng.normal(NOISE_MEAN, NOISE_STD, size=base.shape)
                synthetic = base + noise
                X_sel = np.vstack([X_cls, synthetic])
            else:
                X_sel = X_cls

        new_X.append(X_sel)
        new_y.append(np.full(len(X_sel), lbl_idx, dtype=np.int32))
        print(f'[Info] Class {lbl_name}: original={mask.sum()}, balanced={len(X_sel)}')

    X_bal = np.vstack(new_X)
    y_bal = np.concatenate(new_y)

    perm = rng.permutation(len(X_bal))
    return X_bal[perm], y_bal[perm]


def normalize_features(X):
    """Row-wise L1 normalization (matches tf.keras.utils.normalize(axis=1) in the paper)."""
    norms = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def prepare_external_test(filenames, normalize=True, verbose=True,
                          attack_features=False, reorder=None):
    """Load one or more files as an external (held-out) test set.

    Args:
        attack_features: if True, append the 6 ATTACK-targeted features before normalization
        reorder: if a column-order array is passed, apply it after all other transforms
                 (Must be the SAME order used for training data!)

    Returns (X_test, y_test). No splitting or class balancing — kept as-is to
    measure generalization.
    """
    if verbose:
        print('=' * 60)
        print(f'Loading external test set: {filenames}')
        print('=' * 60)

    X, y = load_moore_files(filenames=filenames)

    if verbose:
        print(f'\n[Info] External test set sample count: {len(X)}')
        for i, name in enumerate(LIST_Y):
            cnt = int((y == i).sum())
            if cnt > 0:
                print(f'   {name}: {cnt}')

    if attack_features:
        from feature_engineering import add_attack_features
        X = add_attack_features(X)
        if verbose:
            print(f'[Info] After ATTACK feature engineering: {X.shape}')

    if normalize:
        X = normalize_features(X)

    if reorder is not None:
        X = X[:, reorder]
        if verbose:
            print(f'[Info] Feature reordering applied')

    return X, y


def prepare_data(use_balance=True, normalize=True, verbose=True,
                 train_files=None, external_test_files=None,
                 attack_features=False, reorder=False):
    """End-to-end data loading pipeline.

    Args:
        train_files: list of files used for training and internal test split.
            Defaults to MOORE_FILES (entry01~entry10).
        external_test_files: if specified, an additional external test set is loaded.
        attack_features: append 6 ATTACK-targeted derived features (port-related)
        reorder: apply correlation-clustering-based feature reordering

    Returns:
        (X_train, X_test, y_train, y_test, reorder_arr)
        reorder_arr is the column order applied (np.arange(D) if reorder=False),
        used so external_test can apply the same transform.
    """
    if verbose:
        print('=' * 60)
        print('Starting to load and preprocess the Moore dataset')
        print('=' * 60)

    X, y = load_moore_files(filenames=train_files)

    if verbose:
        print(f'\n[Info] Total samples loaded: {len(X)}')
        print(f'[Info] Distribution of each class:')
        for i, name in enumerate(LIST_Y):
            cnt = int((y == i).sum())
            print(f'   {name}: {cnt}')

    if use_balance:
        if verbose:
            print('\n[Info] Performing class balancing...')
        X, y = balance_dataset(X, y)

    # ATTACK feature engineering must come before normalization (port values are raw integers)
    if attack_features:
        from feature_engineering import add_attack_features
        X = add_attack_features(X)
        if verbose:
            print(f'\n[Info] After ATTACK feature engineering: {X.shape}')

    if normalize:
        if verbose:
            print('\n[Info] Performing feature normalization...')
        X = normalize_features(X)

    # Feature reordering: run after normalization, based on normalized correlations
    reorder_arr = None
    if reorder:
        from feature_engineering import correlation_reorder
        if verbose:
            print('\n[Info] Computing correlation-clustering feature reordering...')
        reorder_arr = correlation_reorder(X)
        X = X[:, reorder_arr]
        if verbose:
            print(f'[Info] Reorder array length: {len(reorder_arr)}')
    else:
        reorder_arr = np.arange(X.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    if verbose:
        print(f'\n[Info] Training set size: {X_train.shape}, Test set size: {X_test.shape}')
        print('=' * 60)

    return X_train, X_test, y_train, y_test, reorder_arr


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, _ = prepare_data()
    print('\nShape verification:')
    print(f'  X_train: {X_train.shape}')
    print(f'  X_test : {X_test.shape}')
    print(f'  y_train: {y_train.shape}')
    print(f'  y_test : {y_test.shape}')
