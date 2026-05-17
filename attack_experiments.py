"""
ATTACK-class targeted experiments.

Experiment sequence:
  #1a baseline-large-batch: same as the best dilated model from earlier ablation,
       just with a bigger batch (verifies GPU-throughput optimization)
  #1b weighted-CE: weighted Cross-Entropy, ATTACK class weight = 5/10/20
  #1c focal-loss : Focal Loss (gamma=2) + class weights
  #2  two-stage  : binary ATTACK detector + 11-class classifier
  #6  (fallback) : binary ATTACK detector with port columns zeroed out
                   (only used if #2 does not bring enough gain)

Each experiment reports:
  - Internal test set ATTACK precision/recall/F1, plus overall accuracy
  - External test set (entry12) -- but entry12 has 0 ATTACK samples, so it
    only tests whether non-ATTACK classes get hurt
"""
import os
import json
import time
import argparse
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
)
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, LIST_Y, TEST_SIZE, RANDOM_STATE, NUM_CLASSES
from data_preprocess import load_moore_files, balance_dataset, normalize_features
from utils_eval import plot_confusion_matrix, per_class_accuracy
import models_torch as mt


ATTACK_IDX = LIST_Y.index('ATTACK')


def build_dataset():
    """Load and balance the Moore dataset; return the train/test split."""
    X, y = load_moore_files()
    X, y = balance_dataset(X, y)
    X = normalize_features(X)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)


def attack_metrics(y_true, y_pred):
    """Report ATTACK class precision / recall / F1."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[ATTACK_IDX], zero_division=0
    )
    return {
        'attack_precision': float(p[0]),
        'attack_recall': float(r[0]),
        'attack_f1': float(f[0]),
        'overall_accuracy': float(accuracy_score(y_true, y_pred)),
    }


# ============== Experiment #1: weighted CE / Focal Loss ==============

def exp_baseline(X_tr, X_te, y_tr, y_te, epochs, batch_size):
    print('\n## Experiment 1a: baseline (large batch)')
    res = mt.train_cnn1d_dilated(X_tr, y_tr, X_te, y_te,
                                 epochs=epochs, batch_size=batch_size, verbose=0)
    return res


def exp_weighted_ce(X_tr, X_te, y_tr, y_te, epochs, batch_size, attack_weight):
    print(f'\n## Experiment 1b: weighted CE (ATTACK x{attack_weight})')
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[ATTACK_IDX] = attack_weight
    res = mt.train_cnn1d_dilated(X_tr, y_tr, X_te, y_te,
                                 epochs=epochs, batch_size=batch_size, verbose=0,
                                 class_weights=weights)
    return res


def exp_focal_loss(X_tr, X_te, y_tr, y_te, epochs, batch_size, attack_weight):
    print(f'\n## Experiment 1c: focal loss (gamma=2, ATTACK x{attack_weight})')
    weights = np.ones(NUM_CLASSES, dtype=np.float32)
    weights[ATTACK_IDX] = attack_weight
    res = mt.train_cnn1d_dilated(X_tr, y_tr, X_te, y_te,
                                 epochs=epochs, batch_size=batch_size, verbose=0,
                                 class_weights=weights, use_focal=True)
    return res


# ============== Experiment #2: two-stage classifier ==============

def exp_two_stage(X_tr, X_te, y_tr, y_te, epochs, batch_size,
                  binary_attack_weight=20.0, zero_ports=False):
    """Two-stage classifier.

    Stage 1: binary "is ATTACK" classifier (heavily weighted to maximize recall)
    Stage 2: 11-class classifier on non-ATTACK samples

    Final prediction:
        if Stage1 predicts ATTACK -> ATTACK
        else -> Stage2 prediction (with class indices re-mapped back to 12-class)
    """
    print(f'\n## Experiment 2: two-stage (ATTACK weight={binary_attack_weight}, zero_ports={zero_ports})')

    # ---------- Stage 1: binary ATTACK detector ----------
    y_tr_bin = (y_tr == ATTACK_IDX).astype(np.int64)
    y_te_bin = (y_te == ATTACK_IDX).astype(np.int64)

    X_tr_s1 = X_tr.copy()
    X_te_s1 = X_te.copy()
    if zero_ports:
        # Wipe the first 2 columns (= server-port, client-port). Forces the model
        # to learn ATTACK from packet-size / timing / flag features.
        X_tr_s1[:, :2] = 0.0
        X_te_s1[:, :2] = 0.0

    weights = np.array([1.0, binary_attack_weight], dtype=np.float32)
    print('  Training Stage 1 (binary)...')
    s1 = mt.train_cnn1d_dilated(X_tr_s1, y_tr_bin, X_te_s1, y_te_bin,
                                epochs=epochs, batch_size=batch_size, verbose=0,
                                class_weights=weights, num_classes=2)
    s1_pred = s1['pred']
    s1_metrics = {
        'binary_acc': float(s1['accuracy']),
        'binary_recall_attack': float(((s1_pred == 1) & (y_te_bin == 1)).sum() / max(1, (y_te_bin == 1).sum())),
        'binary_precision_attack': float(((s1_pred == 1) & (y_te_bin == 1)).sum() / max(1, (s1_pred == 1).sum())),
    }
    print(f'  Stage 1: acc={s1_metrics["binary_acc"]:.4f}, '
          f'ATTACK recall={s1_metrics["binary_recall_attack"]:.4f}, '
          f'precision={s1_metrics["binary_precision_attack"]:.4f}')

    # ---------- Stage 2: 11-class classifier on non-ATTACK ----------
    # Build remap: original index -> reduced index (skipping ATTACK)
    non_attack_orig = [i for i in range(NUM_CLASSES) if i != ATTACK_IDX]
    orig_to_reduced = {orig: i for i, orig in enumerate(non_attack_orig)}
    reduced_to_orig = {i: orig for orig, i in orig_to_reduced.items()}

    mask_tr = (y_tr != ATTACK_IDX)
    X_tr_s2 = X_tr[mask_tr]
    y_tr_s2 = np.array([orig_to_reduced[int(v)] for v in y_tr[mask_tr]], dtype=np.int64)

    mask_te = (y_te != ATTACK_IDX)
    X_te_s2 = X_te[mask_te]
    y_te_s2 = np.array([orig_to_reduced[int(v)] for v in y_te[mask_te]], dtype=np.int64)

    print('  Training Stage 2 (11-class)...')
    s2 = mt.train_cnn1d_dilated(X_tr_s2, y_tr_s2, X_te_s2, y_te_s2,
                                epochs=epochs, batch_size=batch_size, verbose=0,
                                num_classes=NUM_CLASSES - 1)

    # ---------- Combine the two stages on the full test set ----------
    # Stage 1 prediction on all test samples
    # Stage 2 prediction on all test samples (only used where Stage 1 says non-ATTACK)
    s2_pred_all = mt.predict(s2['model'], X_te)
    s2_pred_orig = np.array([reduced_to_orig[int(v)] for v in s2_pred_all])

    s1_pred_all = mt.predict(s1['model'], X_te_s1)
    final_pred = np.where(s1_pred_all == 1, ATTACK_IDX, s2_pred_orig)

    return {
        'model_stage1': s1['model'],
        'model_stage2': s2['model'],
        'pred': final_pred,
        'accuracy': float(accuracy_score(y_te, final_pred)),
        'time': s1['time'] + s2['time'],
        'stage1_metrics': s1_metrics,
    }


# ============== Main ==============

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch-size', type=int, default=2048,
                   help='Larger batch (2048 by default) to fully utilize GPU')
    p.add_argument('--attack-weight', type=float, default=5.0,
                   help='ATTACK class weight (#1b/#1c)')
    p.add_argument('--binary-attack-weight', type=float, default=20.0,
                   help='ATTACK class weight for the binary stage (#2)')
    p.add_argument('--skip-1a', action='store_true')
    p.add_argument('--skip-1b', action='store_true')
    p.add_argument('--skip-1c', action='store_true')
    p.add_argument('--skip-2', action='store_true')
    p.add_argument('--enable-6', action='store_true',
                   help='Also run #6 (port-zeroed two-stage)')
    args = p.parse_args()

    print(f'[Info] Using GPU: {torch.cuda.is_available()}, device={mt.get_device()}')

    print('\n>>> Building dataset...')
    X_tr, X_te, y_tr, y_te = build_dataset()
    print(f'[Info] Train {X_tr.shape}, Test {X_te.shape}')
    print(f'[Info] ATTACK samples in train: {(y_tr == ATTACK_IDX).sum()} '
          f'({(y_tr == ATTACK_IDX).mean()*100:.2f}%)')
    print(f'[Info] ATTACK samples in test : {(y_te == ATTACK_IDX).sum()}')

    summary = []

    def record(name, res, save_cm=True):
        metrics = attack_metrics(y_te, res['pred'])
        per_cls = per_class_accuracy(y_te, res['pred'])
        entry = {'name': name, **metrics, 'time': res.get('time', 0)}
        if 'stage1_metrics' in res:
            entry['stage1'] = res['stage1_metrics']
        summary.append(entry)
        print(f'[{name}] overall={metrics["overall_accuracy"]:.4f}, '
              f'ATTACK P={metrics["attack_precision"]:.4f}, R={metrics["attack_recall"]:.4f}, '
              f'F1={metrics["attack_f1"]:.4f}, time={res.get("time",0):.1f}s')
        if save_cm:
            plot_confusion_matrix(
                y_te, res['pred'], f'ATTACK Exp: {name}',
                save_path=os.path.join(OUTPUT_DIR, f'attack_exp_{name}.png')
            )

    if not args.skip_1a:
        record('1a_baseline', exp_baseline(X_tr, X_te, y_tr, y_te,
                                           args.epochs, args.batch_size))

    if not args.skip_1b:
        record(f'1b_weighted_ce_w{int(args.attack_weight)}',
               exp_weighted_ce(X_tr, X_te, y_tr, y_te,
                              args.epochs, args.batch_size, args.attack_weight))
        # Also try a larger weight
        record('1b_weighted_ce_w20',
               exp_weighted_ce(X_tr, X_te, y_tr, y_te,
                              args.epochs, args.batch_size, 20.0))

    if not args.skip_1c:
        record(f'1c_focal_w{int(args.attack_weight)}',
               exp_focal_loss(X_tr, X_te, y_tr, y_te,
                             args.epochs, args.batch_size, args.attack_weight))

    if not args.skip_2:
        record('2_two_stage',
               exp_two_stage(X_tr, X_te, y_tr, y_te,
                            args.epochs, args.batch_size,
                            args.binary_attack_weight))

    if args.enable_6:
        record('6_two_stage_zero_ports',
               exp_two_stage(X_tr, X_te, y_tr, y_te,
                            args.epochs, args.batch_size,
                            args.binary_attack_weight, zero_ports=True))

    # ---------- Save report ----------
    out_path = os.path.join(OUTPUT_DIR, 'attack_experiments.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': vars(args),
            'results': summary,
        }, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 90)
    print(' ATTACK EXPERIMENT SUMMARY')
    print('=' * 90)
    print(f'{"Name":<28} {"Overall":<10} {"Attack-P":<10} {"Attack-R":<10} {"Attack-F1":<10} {"Time(s)":<10}')
    print('-' * 90)
    for s in summary:
        print(f'{s["name"]:<28} '
              f'{s["overall_accuracy"]:<10.4f} '
              f'{s["attack_precision"]:<10.4f} '
              f'{s["attack_recall"]:<10.4f} '
              f'{s["attack_f1"]:<10.4f} '
              f'{s["time"]:<10.1f}')
    print('=' * 90)
    print(f'\nSummary saved to {out_path}')


if __name__ == '__main__':
    main()
