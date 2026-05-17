"""
Fair comparison between the paper's improved 2D CNN (Section 5.2) and the
proposed Dilated 1D CNN, both running on the same PyTorch GPU pipeline.

Configurations:
  - paper_orig:     2D CNN, filters=(8,16),  batch=128  (paper baseline)
  - paper_improved: 2D CNN, filters=(16,32), batch=64   (paper Section 5.2 claim: 99.67%)
  - dilated_b128:   Dilated 1D CNN, batch=128
  - dilated_b64:    Dilated 1D CNN, batch=64
"""
import os
import json
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, LIST_Y, TEST_SIZE, RANDOM_STATE
from data_preprocess import load_moore_files, balance_dataset, normalize_features
from utils_eval import per_class_accuracy
import models_torch as mt


ATTACK_IDX = LIST_Y.index('ATTACK')


def main():
    print('>>> Building dataset...')
    X, y = load_moore_files()
    X, y = balance_dataset(X, y)
    X = normalize_features(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f'Train {X_tr.shape}, Test {X_te.shape}')
    print(f'Device: {mt.get_device()}')

    configs = [
        # (name, model_fn, kwargs, batch)
        ('paper_orig',     mt.train_cnn,           dict(conv_filters=(8, 16)),  128),
        ('paper_improved', mt.train_cnn,           dict(conv_filters=(16, 32)), 64),
        ('dilated_b128',   mt.train_cnn1d_dilated, dict(),                      128),
        ('dilated_b64',    mt.train_cnn1d_dilated, dict(),                      64),
    ]

    EPOCHS = 25
    results = []

    for name, fn, kwargs, bs in configs:
        print(f'\n=== {name} (batch={bs}, epochs={EPOCHS}) ===')
        res = fn(X_tr, y_tr, X_te, y_te, epochs=EPOCHS, batch_size=bs, verbose=0, **kwargs)
        pred = res['pred']
        p, r, f, _ = precision_recall_fscore_support(
            y_te, pred, labels=[ATTACK_IDX], zero_division=0)
        acc = accuracy_score(y_te, pred)
        per_cls = per_class_accuracy(y_te, pred)
        entry = {
            'config': name,
            'batch_size': bs,
            'overall_accuracy': float(acc),
            'attack_precision': float(p[0]),
            'attack_recall': float(r[0]),
            'attack_f1': float(f[0]),
            'time': float(res['time']),
            'per_class_accuracy': per_cls,
        }
        results.append(entry)
        print(f'  Overall {acc:.4f} | ATTACK P={p[0]:.4f} R={r[0]:.4f} F1={f[0]:.4f}')

    # Save & report
    out_path = os.path.join(OUTPUT_DIR, 'final_comparison.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 90)
    print(' FINAL COMPARISON ')
    print('=' * 90)
    print(f'{"Config":<18} {"Batch":<7} {"Overall":<10} {"ATK-P":<10} {"ATK-R":<10} {"ATK-F1":<10} {"Time(s)":<10}')
    print('-' * 90)
    for r in results:
        print(f'{r["config"]:<18} {r["batch_size"]:<7} '
              f'{r["overall_accuracy"]:<10.4f} '
              f'{r["attack_precision"]:<10.4f} '
              f'{r["attack_recall"]:<10.4f} '
              f'{r["attack_f1"]:<10.4f} '
              f'{r["time"]:<10.1f}')
    print('=' * 90)
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
