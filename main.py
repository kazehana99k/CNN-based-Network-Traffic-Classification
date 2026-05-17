"""
Entry script: run six algorithms and compare them.
The user can specify which algorithms to run via command-line arguments.

Usage examples:
    python main.py --all                       # Run all six algorithms
    python main.py --cnn                       # Run only CNN
    python main.py --cnn --bp                  # Run only CNN and BP
    python main.py --epochs 25 --batch-size 128
"""
import os
import argparse
import json
import time

import numpy as np

from config import OUTPUT_DIR, LIST_Y, EPOCHS, BATCH_SIZE
from data_preprocess import prepare_data, prepare_external_test
from utils_eval import (
    plot_confusion_matrix, per_class_accuracy,
    plot_per_class_accuracy, plot_training_history,
    plot_overall_comparison, print_classification_summary,
)


def _predict_with_model(model_name, fitted_model, X_ext):
    """Run prediction on the external test set for a trained model.

    Different algorithms need different input shapes (CNN 4D / CNN1D 3D / others 2D).
    """
    import tensorflow as tf
    import numpy as np
    if model_name == 'CNN':
        X_in = tf.reshape(tf.cast(X_ext, tf.float32), [-1, 16, 16, 1])
        return np.argmax(fitted_model.predict(X_in, verbose=0), axis=1)
    if model_name in ('CNN1D', 'CNN1D-Dilated'):
        X_in = tf.reshape(tf.cast(X_ext, tf.float32), [-1, X_ext.shape[1], 1])
        return np.argmax(fitted_model.predict(X_in, verbose=0), axis=1)
    if model_name == 'BP':
        X_in = tf.cast(X_ext, tf.float32)
        return np.argmax(fitted_model.predict(X_in, verbose=0), axis=1)
    return fitted_model.predict(X_ext)


def parse_args():
    p = argparse.ArgumentParser(description='Moore traffic classification - reproduce CNN + 5 comparison algorithms')
    p.add_argument('--all', action='store_true', help='Run all algorithms')
    p.add_argument('--cnn', action='store_true', help='Run 2D CNN (paper baseline)')
    p.add_argument('--cnn1d', action='store_true', help='Run 1D CNN (treats features as a sequence)')
    p.add_argument('--cnn1d-dilated', dest='cnn1d_dilated', action='store_true',
                   help='Run dilated 1D CNN (large receptive field, fewer params)')
    p.add_argument('--bp', action='store_true', help='Run BP neural network')
    p.add_argument('--knn', action='store_true', help='Run KNN')
    p.add_argument('--nb', action='store_true', help='Run Naive Bayes')
    p.add_argument('--svm', action='store_true', help='Run SVM (note: slow on large data)')
    p.add_argument('--dt', action='store_true', help='Run Decision Tree')

    p.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs (CNN/BP)')
    p.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='batch size')
    p.add_argument('--no-balance', action='store_true', help='Disable class balancing')
    p.add_argument('--no-normalize', action='store_true', help='Disable feature normalization')
    p.add_argument('--svm-subset', type=int, default=20000,
                   help='Subset size used for SVM training (avoid being too slow); set to 0 to use the full training set')
    p.add_argument('--external-test', nargs='+', default=None,
                   help='External test set file names (placed under data/moore/), e.g. entry12.weka.allclass.arff. '
                        'Used to evaluate model generalization, not used for training.')
    p.add_argument('--cnn1d-kernel', type=int, default=5, help='1D CNN kernel size')
    p.add_argument('--cnn1d-pool', type=int, default=4, help='1D CNN pooling stride')
    p.add_argument('--reorder', action='store_true',
                   help='Reorder features via correlation clustering (let semantically similar features sit adjacent)')
    p.add_argument('--attack-features', dest='attack_features', action='store_true',
                   help='Add 6 ATTACK-targeted derived features (port indicators / log-transformations)')
    return p.parse_args()


def select_algorithms(args):
    """Decide which algorithms to run based on the arguments."""
    all_algos = ['cnn', 'cnn1d', 'cnn1d_dilated', 'bp', 'knn', 'nb', 'svm', 'dt']
    selected = []
    if args.all:
        selected = list(all_algos)
    else:
        for name in all_algos:
            if getattr(args, name):
                selected.append(name)
    if not selected:
        selected = ['cnn']  # default to CNN if not specified
    return selected


def run_cnn(X_train, y_train, X_test, y_test, args, summary, results):
    if X_train.shape[1] != 256:
        print(f'\n[Warn] 2D CNN requires feature dim == 256 but got {X_train.shape[1]} '
              f'(probably because --attack-features adds 6 features). Skipping 2D CNN.')
        return
    from models_dl import train_cnn
    print('\n' + '#' * 60)
    print('# Running CNN')
    print('#' * 60)
    res = train_cnn(X_train, y_train, X_test, y_test,
                    epochs=args.epochs, batch_size=args.batch_size)
    summary['CNN'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['CNN'] = res

    plot_confusion_matrix(y_test, res['pred'], 'CNN Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn_confusion.png'))
    plot_training_history(res['history'], 'CNN',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn_history.png'))


def run_cnn1d(X_train, y_train, X_test, y_test, args, summary, results):
    from models_dl import train_cnn1d
    print('\n' + '#' * 60)
    print('# Running 1D CNN')
    print('#' * 60)
    res = train_cnn1d(X_train, y_train, X_test, y_test,
                     epochs=args.epochs, batch_size=args.batch_size,
                     kernel_size=args.cnn1d_kernel, pool_size=args.cnn1d_pool)
    summary['CNN1D'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['CNN1D'] = res

    plot_confusion_matrix(y_test, res['pred'], '1D CNN Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn1d_confusion.png'))
    plot_training_history(res['history'], '1D CNN',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn1d_history.png'))


def run_cnn1d_dilated(X_train, y_train, X_test, y_test, args, summary, results):
    from models_dl import train_cnn1d_dilated
    print('\n' + '#' * 60)
    print('# Running Dilated 1D CNN')
    print('#' * 60)
    res = train_cnn1d_dilated(X_train, y_train, X_test, y_test,
                              epochs=args.epochs, batch_size=args.batch_size)
    summary['CNN1D-Dilated'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['CNN1D-Dilated'] = res

    plot_confusion_matrix(y_test, res['pred'], 'Dilated 1D CNN Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn1d_dilated_confusion.png'))
    plot_training_history(res['history'], 'Dilated 1D CNN',
                          save_path=os.path.join(OUTPUT_DIR, 'cnn1d_dilated_history.png'))


def run_bp(X_train, y_train, X_test, y_test, args, summary, results):
    from models_dl import train_bp
    print('\n' + '#' * 60)
    print('# Running BP Neural Network')
    print('#' * 60)
    res = train_bp(X_train, y_train, X_test, y_test,
                   epochs=args.epochs, batch_size=args.batch_size)
    summary['BP'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['BP'] = res

    plot_confusion_matrix(y_test, res['pred'], 'BP Neural Network Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'bp_confusion.png'))
    plot_training_history(res['history'], 'BP',
                          save_path=os.path.join(OUTPUT_DIR, 'bp_history.png'))


def run_knn(X_train, y_train, X_test, y_test, args, summary, results):
    from models_ml import train_knn
    print('\n' + '#' * 60)
    print('# Running KNN')
    print('#' * 60)
    res = train_knn(X_train, y_train, X_test, y_test)
    summary['KNN'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['KNN'] = res
    plot_confusion_matrix(y_test, res['pred'], 'KNN Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'knn_confusion.png'))


def run_nb(X_train, y_train, X_test, y_test, args, summary, results):
    from models_ml import train_naive_bayes
    print('\n' + '#' * 60)
    print('# Running Naive Bayes')
    print('#' * 60)
    res = train_naive_bayes(X_train, y_train, X_test, y_test)
    summary['NaiveBayes'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['NaiveBayes'] = res
    plot_confusion_matrix(y_test, res['pred'], 'Naive Bayes Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'nb_confusion.png'))


def run_svm(X_train, y_train, X_test, y_test, args, summary, results):
    from models_ml import train_svm
    print('\n' + '#' * 60)
    print('# Running SVM')
    print('#' * 60)

    if args.svm_subset and args.svm_subset > 0 and args.svm_subset < len(X_train):
        rng = np.random.default_rng(0)
        idx = rng.choice(len(X_train), args.svm_subset, replace=False)
        Xt, yt = X_train[idx], y_train[idx]
        print(f'[SVM] Using {args.svm_subset} samples for training (avoid being too slow).')
    else:
        Xt, yt = X_train, y_train

    res = train_svm(Xt, yt, X_test, y_test)
    summary['SVM'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['SVM'] = res
    plot_confusion_matrix(y_test, res['pred'], 'SVM Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'svm_confusion.png'))


def run_dt(X_train, y_train, X_test, y_test, args, summary, results):
    from models_ml import train_decision_tree
    print('\n' + '#' * 60)
    print('# Running Decision Tree')
    print('#' * 60)
    res = train_decision_tree(X_train, y_train, X_test, y_test)
    summary['DecisionTree'] = {'accuracy': res['accuracy'], 'time': res['time']}
    results['DecisionTree'] = res
    plot_confusion_matrix(y_test, res['pred'], 'Decision Tree Confusion Matrix',
                          save_path=os.path.join(OUTPUT_DIR, 'dt_confusion.png'))


def main():
    args = parse_args()
    selected = select_algorithms(args)
    print(f'[Info] Selected algorithms: {selected}')
    print(f'[Info] epochs={args.epochs}, batch_size={args.batch_size}')

    # 1. Load data
    X_train, X_test, y_train, y_test, reorder_arr = prepare_data(
        use_balance=not args.no_balance,
        normalize=not args.no_normalize,
        attack_features=args.attack_features,
        reorder=args.reorder,
    )

    feature_dim = X_train.shape[1]
    print(f'[Info] Final feature dim: {feature_dim}')
    if feature_dim != 256:
        print(f'[Info] Note: feature_dim != 256, 2D CNN (16x16) will be disabled')

    # 2. Run the selected algorithms
    summary = {}      # accuracy / time
    results = {}      # full results
    per_class_dict = {}  # per-class accuracy

    runners = {
        'cnn': run_cnn,
        'cnn1d': run_cnn1d,
        'cnn1d_dilated': run_cnn1d_dilated,
        'bp': run_bp,
        'knn': run_knn,
        'nb': run_nb,
        'svm': run_svm,
        'dt': run_dt,
    }

    for algo in selected:
        runners[algo](X_train, y_train, X_test, y_test, args, summary, results)
        # Compute per-class accuracy
        algo_name = list(summary.keys())[-1]
        per_class_dict[algo_name] = per_class_accuracy(y_test, results[algo_name]['pred'])

    # 2.5 Optional: evaluate on the external test set
    external_summary = {}
    external_per_class = {}
    if args.external_test:
        print('\n' + '#' * 60)
        print(f'# External test set evaluation: {args.external_test}')
        print('#' * 60)
        X_ext, y_ext = prepare_external_test(
            args.external_test,
            normalize=not args.no_normalize,
            attack_features=args.attack_features,
            reorder=reorder_arr if args.reorder else None,
        )

        from sklearn.metrics import accuracy_score
        for algo_name, res in results.items():
            try:
                ext_pred = _predict_with_model(algo_name, res['model'], X_ext)
                acc = accuracy_score(y_ext, ext_pred)
                external_summary[algo_name] = {'accuracy': acc}
                external_per_class[algo_name] = per_class_accuracy(y_ext, ext_pred)
                print(f'  [{algo_name}] External test accuracy: {acc:.4f}')
                plot_confusion_matrix(
                    y_ext, ext_pred,
                    f'{algo_name} External Test (entry12)',
                    save_path=os.path.join(OUTPUT_DIR, f'{algo_name.lower()}_external_confusion.png')
                )
            except Exception as e:
                print(f'  [{algo_name}] External evaluation failed: {e}')

    # 3. Generate the comprehensive comparison chart
    print('\n' + '#' * 60)
    print('# Generating overall comparison report')
    print('#' * 60)

    if summary:
        plot_overall_comparison(summary, save_path=os.path.join(OUTPUT_DIR, 'overall_comparison.png'))

    if per_class_dict:
        plot_per_class_accuracy(per_class_dict,
                                save_path=os.path.join(OUTPUT_DIR, 'per_class_accuracy.png'))

    # 4. Save the JSON summary
    summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': summary,
            'per_class_accuracy': per_class_dict,
            'external_summary': external_summary,
            'external_per_class': external_per_class,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'balanced': not args.no_balance,
                'normalized': not args.no_normalize,
                'external_test_files': args.external_test,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f'\n[Info] Summary saved to {summary_path}')

    # 5. Print the final summary
    print('\n' + '=' * 60)
    print(' FINAL SUMMARY ')
    print('=' * 60)
    print(f'{"Algorithm":<20} {"Accuracy":<12} {"Time (s)":<12}')
    print('-' * 50)
    for name, info in summary.items():
        print(f'{name:<20} {info["accuracy"]:<12.4f} {info["time"]:<12.2f}')
    print('=' * 60)

    if external_summary:
        print('\n EXTERNAL TEST (no training on these samples) ')
        print('=' * 60)
        print(f'{"Algorithm":<20} {"Ext. Accuracy":<14}')
        print('-' * 50)
        for name, info in external_summary.items():
            print(f'{name:<20} {info["accuracy"]:<14.4f}')
        print('=' * 60)


if __name__ == '__main__':
    main()
