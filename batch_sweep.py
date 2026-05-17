"""Batch-size sweep: find the optimal batch for ATTACK F1 (Dilated CNN1D + plain CE)."""
import os, json, argparse
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split

from config import OUTPUT_DIR, LIST_Y, TEST_SIZE, RANDOM_STATE
from data_preprocess import load_moore_files, balance_dataset, normalize_features
import models_torch as mt

ATTACK_IDX = LIST_Y.index('ATTACK')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batches', type=int, nargs='+', default=[128, 256, 512, 1024, 2048])
    args = p.parse_args()

    print('>>> Building dataset...')
    X, y = load_moore_files()
    X, y = balance_dataset(X, y)
    X = normalize_features(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f'Train {X_tr.shape}, Test {X_te.shape}')

    results = []
    for bs in args.batches:
        print(f'\n=== Batch size {bs} ===')
        res = mt.train_cnn1d_dilated(X_tr, y_tr, X_te, y_te,
                                     epochs=args.epochs, batch_size=bs, verbose=0)
        pred = res['pred']
        p_, r_, f_, _ = precision_recall_fscore_support(
            y_te, pred, labels=[ATTACK_IDX], zero_division=0)
        acc = accuracy_score(y_te, pred)
        entry = {
            'batch': bs,
            'overall_accuracy': float(acc),
            'attack_precision': float(p_[0]),
            'attack_recall': float(r_[0]),
            'attack_f1': float(f_[0]),
            'time': float(res['time']),
        }
        results.append(entry)
        print(f'  Overall {acc:.4f} | ATTACK P={p_[0]:.4f} R={r_[0]:.4f} F1={f_[0]:.4f} | {res["time"]:.1f}s')

    # Save and report
    out = os.path.join(OUTPUT_DIR, 'batch_sweep.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 75)
    print(' BATCH SIZE SWEEP - ATTACK F1 CURVE ')
    print('=' * 75)
    print(f'{"Batch":<8} {"Overall":<10} {"ATK-P":<10} {"ATK-R":<10} {"ATK-F1":<10} {"Time(s)":<10}')
    print('-' * 75)
    for r in results:
        print(f'{r["batch"]:<8} {r["overall_accuracy"]:<10.4f} '
              f'{r["attack_precision"]:<10.4f} {r["attack_recall"]:<10.4f} '
              f'{r["attack_f1"]:<10.4f} {r["time"]:<10.1f}')
    print('=' * 75)

    best = max(results, key=lambda x: x['attack_f1'])
    print(f'\nBest batch by ATTACK F1: {best["batch"]} (F1={best["attack_f1"]:.4f})')

    # Plot
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        bs_list = [r['batch'] for r in results]
        fig, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(bs_list, [r['attack_f1'] for r in results], 'o-', label='ATTACK F1', color='tab:red')
        ax1.plot(bs_list, [r['attack_precision'] for r in results], 's--', label='ATTACK P', color='tab:orange')
        ax1.plot(bs_list, [r['attack_recall'] for r in results], '^--', label='ATTACK R', color='tab:purple')
        ax1.plot(bs_list, [r['overall_accuracy'] for r in results], 'd-', label='Overall Acc', color='tab:blue')
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(bs_list)
        ax1.set_xticklabels([str(b) for b in bs_list])
        ax1.set_xlabel('Batch size (log scale)')
        ax1.set_ylabel('Accuracy / F1')
        ax1.set_title('Batch Size vs ATTACK F1 (Dilated CNN1D)')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='lower left')
        ax1.set_ylim(0, 1.02)
        out_png = os.path.join(OUTPUT_DIR, 'batch_sweep.png')
        plt.tight_layout()
        plt.savefig(out_png, dpi=120)
        print(f'Chart saved to {out_png}')
    except Exception as e:
        print(f'Plot failed: {e}')


if __name__ == '__main__':
    main()
