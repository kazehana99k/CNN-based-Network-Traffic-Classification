"""
Ablation study: compare the contribution of each 1D CNN improvement.

Configurations:
  1. baseline: vanilla 1D CNN (k=5)
  2. +k9: kernel 9 (larger receptive field)
  3. +reorder: feature reordering via correlation clustering
  4. +attack: 6 extra port-aware features
  5. +dilated: dilated 1D CNN (k=3, dilations 1/2/4)
  6. all: 1D CNN k=9 + reorder + attack features

The data is loaded ONCE, then variants are derived from it (avoids the minute-scale Moore parsing).
Each variant is also evaluated on the entry12 external test for generalization comparison.
"""
import os
import json
import time
import argparse
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from config import (
    MOORE_FILES, OUTPUT_DIR, LIST_Y, TEST_SIZE, RANDOM_STATE,
    EPOCHS, BATCH_SIZE,
)
from data_preprocess import (
    load_moore_files, balance_dataset, normalize_features,
)
from feature_engineering import add_attack_features, correlation_reorder
from utils_eval import plot_confusion_matrix, per_class_accuracy

# PyTorch backend (GPU-accelerated)
import models_torch as mt


def build_dataset(use_attack=False, use_reorder=False, verbose=True):
    """Load Moore dataset and apply the specified transforms."""
    X_raw, y = load_moore_files()
    if verbose:
        print(f'[Info] Loaded {len(X_raw)} raw samples')

    X_bal, y_bal = balance_dataset(X_raw, y)
    if verbose:
        print(f'[Info] After balancing: {X_bal.shape}')

    if use_attack:
        X_bal = add_attack_features(X_bal)
        if verbose:
            print(f'[Info] After ATTACK features: {X_bal.shape}')

    X_norm = normalize_features(X_bal)

    if use_reorder:
        if verbose:
            print(f'[Info] Computing correlation reorder...')
        order = correlation_reorder(X_norm)
        X_norm = X_norm[:, order]
    else:
        order = np.arange(X_norm.shape[1])

    X_train, X_test, y_train, y_test = train_test_split(
        X_norm, y_bal, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_bal
    )
    if verbose:
        print(f'[Info] Train {X_train.shape}, Test {X_test.shape}')

    return X_train, X_test, y_train, y_test, order


def build_external_set(order, use_attack=False, use_reorder=False, verbose=True):
    """Load entry12 with the SAME transforms applied for fair external evaluation."""
    if verbose:
        print(f'\n[Info] Loading external test set entry12...')
    X_ext, y_ext = load_moore_files(filenames=['entry12.weka.allclass.arff'])

    if use_attack:
        X_ext = add_attack_features(X_ext)

    X_ext = normalize_features(X_ext)

    if use_reorder:
        X_ext = X_ext[:, order]

    if verbose:
        print(f'[Info] External test set: {X_ext.shape}')
    return X_ext, y_ext


def run_one(config_name, X_train, X_test, y_train, y_test,
            X_ext, y_ext, epochs, batch_size,
            cnn1d_kernel=5, use_dilated=False):
    """Train a single 1D CNN configuration and evaluate (PyTorch / GPU)."""
    print('\n' + '=' * 60)
    print(f'Configuration: {config_name}  (device={mt.get_device()})')
    print('=' * 60)

    if use_dilated:
        res = mt.train_cnn1d_dilated(X_train, y_train, X_test, y_test,
                                     epochs=epochs, batch_size=batch_size, verbose=0)
    else:
        res = mt.train_cnn1d(X_train, y_train, X_test, y_test,
                             epochs=epochs, batch_size=batch_size, verbose=0,
                             kernel_size=cnn1d_kernel, pool_size=4)

    int_acc = res['accuracy']
    int_time = res['time']

    # External test
    pred_ext = mt.predict(res['model'], X_ext)
    ext_acc = accuracy_score(y_ext, pred_ext)

    int_per_cls = per_class_accuracy(y_test, res['pred'])
    ext_per_cls = per_class_accuracy(y_ext, pred_ext)

    # Save confusion matrices
    suffix = config_name.replace(' ', '_').replace('+', 'p')
    plot_confusion_matrix(
        y_test, res['pred'], f'1D CNN [{config_name}] - internal',
        save_path=os.path.join(OUTPUT_DIR, f'ablation_{suffix}_internal.png'),
    )
    plot_confusion_matrix(
        y_ext, pred_ext, f'1D CNN [{config_name}] - external (entry12)',
        save_path=os.path.join(OUTPUT_DIR, f'ablation_{suffix}_external.png'),
    )

    print(f'[{config_name}] internal acc={int_acc:.4f}, external acc={ext_acc:.4f}, time={int_time:.1f}s')
    return {
        'config': config_name,
        'internal_accuracy': float(int_acc),
        'external_accuracy': float(ext_acc),
        'time': float(int_time),
        'internal_per_class': int_per_cls,
        'external_per_class': ext_per_cls,
        # Important per-class accuracy on ATTACK
        'attack_internal': int_per_cls.get('ATTACK', 0.0),
        'attack_external': ext_per_cls.get('ATTACK', 0.0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    p.add_argument('--skip', nargs='*', default=[],
                   help='Configurations to skip (baseline / k9 / reorder / attack / dilated / all)')
    args = p.parse_args()

    results = []
    overall_t1 = time.time()

    # =========== Build dataset for each variant ===========
    # Strategy: prepare 4 dataset variants based on the (attack, reorder) combinations
    # Most efficient: cache per (attack, reorder) combination
    datasets = {}

    def get_dataset(use_attack, use_reorder):
        key = (use_attack, use_reorder)
        if key not in datasets:
            print(f'\n>>> Building dataset variant: attack={use_attack}, reorder={use_reorder} <<<')
            X_tr, X_te, y_tr, y_te, order = build_dataset(
                use_attack=use_attack, use_reorder=use_reorder)
            X_ext, y_ext = build_external_set(order, use_attack=use_attack, use_reorder=use_reorder)
            datasets[key] = (X_tr, X_te, y_tr, y_te, X_ext, y_ext)
        return datasets[key]

    # =========== Configurations to run ===========
    configs = [
        # (name, attack, reorder, dilated, kernel)
        ('baseline', False, False, False, 5),
        ('k9', False, False, False, 9),
        ('reorder', False, True, False, 5),
        ('attack', True, False, False, 5),
        ('dilated', False, False, True, 3),
        ('all', True, True, False, 9),
    ]

    for name, use_attack, use_reorder, use_dilated, kernel in configs:
        if name in args.skip:
            print(f'\n[Skip] {name}')
            continue
        X_tr, X_te, y_tr, y_te, X_ext, y_ext = get_dataset(use_attack, use_reorder)
        res = run_one(name, X_tr, X_te, y_tr, y_te, X_ext, y_ext,
                      args.epochs, args.batch_size,
                      cnn1d_kernel=kernel, use_dilated=use_dilated)
        results.append(res)

    # =========== Save ablation report ===========
    out_path = os.path.join(OUTPUT_DIR, 'ablation_summary.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'results': results,
            'config': {'epochs': args.epochs, 'batch_size': args.batch_size},
            'total_time': time.time() - overall_t1,
        }, f, indent=2, ensure_ascii=False)

    # =========== Print summary table ===========
    print('\n' + '=' * 80)
    print(' ABLATION SUMMARY ')
    print('=' * 80)
    print(f'{"Config":<14} {"Internal":<12} {"External":<12} {"ATTACK-int":<12} {"ATTACK-ext":<12} {"Time(s)":<10}')
    print('-' * 80)
    for r in results:
        print(f'{r["config"]:<14} '
              f'{r["internal_accuracy"]:<12.4f} '
              f'{r["external_accuracy"]:<12.4f} '
              f'{r["attack_internal"]:<12.4f} '
              f'{r["attack_external"]:<12.4f} '
              f'{r["time"]:<10.1f}')
    print('=' * 80)
    print(f'\nAblation summary saved to {out_path}')


if __name__ == '__main__':
    main()
