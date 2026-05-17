"""
Evaluation and visualization utilities: confusion matrix, accuracy curve, per-class accuracy bar chart, etc.
Corresponds to paper Section 4.5 (evaluation metrics) and Figure 4.9 (confusion matrix function code).
"""
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-GUI backend, suitable for batch saving images
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from config import LIST_Y, OUTPUT_DIR


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix',
                          labels=None, save_path=None, normalize=True):
    """Plot a confusion matrix.

    Corresponds to the plot_confusion_matrix function in paper Figure 4.9.
    """
    labels = labels or LIST_Y
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    if normalize:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1, row_sum)
        cm_norm = cm.astype('float') / row_sum
    else:
        cm_norm = cm

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)
    ax.set_ylabel('True')
    ax.set_xlabel('Predicted')

    # Annotate cells
    thresh = cm_norm.max() / 2.0
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f'{cm_norm[i, j]:.2f}',
                    ha='center', va='center',
                    color='white' if cm_norm[i, j] > thresh else 'black',
                    fontsize=7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'[Info] Confusion matrix saved to {save_path}')
    plt.close(fig)
    return cm


def per_class_accuracy(y_true, y_pred, labels=None):
    """Compute the per-class classification accuracy."""
    labels = labels or LIST_Y
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    per_class = {}
    for i, name in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i, i]
        per_class[name] = (correct / total) if total > 0 else 0.0
    return per_class


def plot_per_class_accuracy(results_dict, save_path=None):
    """Plot the per-class accuracy comparison bar chart for multiple algorithms.

    Args:
        results_dict: {algorithm name: {class name: accuracy}}
    """
    labels = LIST_Y
    algos = list(results_dict.keys())
    n_algo = len(algos)
    n_lbl = len(labels)

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.8 / n_algo
    x = np.arange(n_lbl)

    for i, algo in enumerate(algos):
        accs = [results_dict[algo].get(lbl, 0.0) for lbl in labels]
        ax.bar(x + i * width, accs, width, label=algo)

    ax.set_xticks(x + width * (n_algo - 1) / 2)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Accuracy')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'[Info] Per-class accuracy bar chart saved to {save_path}')
    plt.close(fig)


def plot_training_history(history, title='Training History', save_path=None):
    """Plot training accuracy / validation accuracy / loss curves (for CNN, BP)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get('accuracy', []), label='train')
    axes[0].plot(history.get('val_accuracy', []), label='val')
    axes[0].set_title(f'{title} - Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.get('loss', []), label='train')
    axes[1].plot(history.get('val_loss', []), label='val')
    axes[1].set_title(f'{title} - Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'[Info] Training history curve saved to {save_path}')
    plt.close(fig)


def plot_overall_comparison(summary, save_path=None):
    """Plot the bar chart comparing accuracy and time spent across multiple algorithms.

    summary: dict of {algorithm_name: {'accuracy': float, 'time': float}}
    """
    names = list(summary.keys())
    accs = [summary[n]['accuracy'] for n in names]
    times = [summary[n]['time'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(names, accs, color='steelblue')
    axes[0].set_title('Algorithm Accuracy Comparison')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim(0, 1.05)
    for i, a in enumerate(accs):
        axes[0].text(i, a + 0.01, f'{a:.4f}', ha='center')

    axes[1].bar(names, times, color='coral')
    axes[1].set_title('Algorithm Time Comparison')
    axes[1].set_ylabel('Time (s)')
    for i, t in enumerate(times):
        axes[1].text(i, t + max(times) * 0.01, f'{t:.1f}', ha='center')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f'[Info] Comprehensive comparison chart saved to {save_path}')
    plt.close(fig)


def print_classification_summary(y_true, y_pred, labels=None):
    """Print detailed classification report (precision/recall/F1)."""
    labels = labels or LIST_Y
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        zero_division=0,
        digits=4,
    )
    print(report)
    return report
