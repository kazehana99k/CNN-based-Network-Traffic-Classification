"""
PyTorch implementation - GPU accelerated versions for CNN / CNN1D / Dilated CNN1D.

Includes the fixes:
1. Dilated CNN uses MaxPool instead of GlobalAveragePooling to retain detail.
2. Auto-detect GPU; falls back to CPU if not available.
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from config import NUM_CLASSES, EPOCHS, BATCH_SIZE


# ============== Loss functions ==============

class FocalLoss(nn.Module):
    """Focal Loss = (1-pt)^gamma * CE_loss.

    Used to handle class imbalance. The (1-pt)^gamma factor amplifies the loss
    weight on hard examples while suppressing easy ones, perfect for minority
    classes like ATTACK.

    Args:
        alpha: class weights (Tensor of shape (num_classes,)) - None means equal
        gamma: focusing parameter; 2.0 is recommended in the original paper
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============== Model definitions ==============

class CNN2D(nn.Module):
    """2D CNN equivalent to the Keras version (16x16 input)."""
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, ceil_mode=True),
        )
        # After 2x2 -> 4x4, channels 16 -> flatten to 256
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN1D(nn.Module):
    """1D CNN equivalent to the Keras version."""
    def __init__(self, input_dim=256, num_classes=NUM_CLASSES,
                 conv_filters=(8, 16), kernel_size=5, pool_size=4):
        super().__init__()
        pad = kernel_size // 2
        self.features = nn.Sequential(
            nn.Conv1d(1, conv_filters[0], kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Conv1d(conv_filters[0], conv_filters[1], kernel_size, padding=pad), nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size, ceil_mode=True),
        )
        # Estimate output length: input_dim -> ceil(input_dim/pool) -> ceil(.../pool)
        import math
        L = math.ceil(math.ceil(input_dim / pool_size) / pool_size)
        flat = conv_filters[1] * L
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class CNN1DDilated(nn.Module):
    """Dilated 1D CNN — FIXED version.

    Old design: GlobalAveragePooling lost too much detail, caused 82% acc.
    Fix: insert MaxPool1d between dilated convs to gradually reduce sequence
    length, then Flatten -> Dense (same as ordinary CNN1D), retaining details
    while still benefitting from larger receptive field.
    """
    def __init__(self, input_dim=256, num_classes=NUM_CLASSES,
                 channels=(16, 32, 32), kernel_size=3,
                 dilations=(1, 2, 4), pool_size=2):
        super().__init__()
        pad_list = [d * (kernel_size // 2) for d in dilations]
        layers_list = []
        in_c = 1
        for f, d, p in zip(channels, dilations, pad_list):
            layers_list += [
                nn.Conv1d(in_c, f, kernel_size, padding=p, dilation=d),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(pool_size, ceil_mode=True),
            ]
            in_c = f
        self.features = nn.Sequential(*layers_list)

        # Estimate output length: divide by pool 3 times
        import math
        L = input_dim
        for _ in range(len(channels)):
            L = math.ceil(L / pool_size)
        flat = channels[-1] * L
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ============== Training routines ==============

def _train_loop(model, X_train, y_train, X_test, y_test,
                epochs, batch_size, lr=1e-3, device=None, verbose=1,
                loss_fn=None, class_weights=None, use_focal=False, focal_gamma=2.0):
    """GPU-optimized training loop.

    Optimizations vs the previous version:
    1. Preload the entire train+test set to GPU once (~150 MB total, fits in 5090's 32GB)
    2. Use a fixed index permutation each epoch + GPU-side slicing (no CPU-GPU transfer overhead)
    3. Larger batch (recommended >=512) amortizes kernel-launch overhead

    loss_fn priority:
        explicit loss_fn arg  >  use_focal  >  class_weights (CE)  >  plain CE
    """
    device = device or get_device()
    model.to(device)

    # Preload entire dataset onto the GPU
    X_train_g = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y_train_g = torch.as_tensor(y_train, dtype=torch.long, device=device)
    X_test_g = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    y_test_g = torch.as_tensor(y_test, dtype=torch.long, device=device)

    # Build loss function
    if loss_fn is None:
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        else:
            cw = None
        if use_focal:
            loss_fn = FocalLoss(alpha=cw, gamma=focal_gamma)
        else:
            loss_fn = nn.CrossEntropyLoss(weight=cw)

    opt = optim.Adam(model.parameters(), lr=lr)
    n_train = len(X_train_g)

    history = {'accuracy': [], 'loss': [], 'val_accuracy': [], 'val_loss': []}

    t1 = time.time()
    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        ep_correct = 0

        # Build random permutation on GPU
        perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_g[idx]
            yb = y_train_g[idx]
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
            ep_correct += (logits.argmax(1) == yb).sum().item()
        train_acc = ep_correct / n_train
        train_loss = ep_loss / n_train

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_test_g)
            val_loss = loss_fn(val_logits, y_test_g).item()
            val_acc = (val_logits.argmax(1) == y_test_g).float().mean().item()

        history['accuracy'].append(train_acc)
        history['loss'].append(train_loss)
        history['val_accuracy'].append(val_acc)
        history['val_loss'].append(val_loss)
        if verbose:
            print(f'Epoch {ep+1}/{epochs}: loss={train_loss:.4f} acc={train_acc:.4f}  '
                  f'val_loss={val_loss:.4f} val_acc={val_acc:.4f}')

    elapsed = time.time() - t1

    # Final eval & prediction
    model.eval()
    with torch.no_grad():
        logits = model(X_test_g)
        pred = logits.argmax(1).cpu().numpy()
        final_acc = float((pred == y_test).mean())

    return {
        'model': model,
        'history': history,
        'accuracy': final_acc,
        'loss': float(history['val_loss'][-1]),
        'time': elapsed,
        'pred': pred,
    }


def train_cnn(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1):
    # X has shape (N, 256); reshape to (N, 1, 16, 16)
    X_tr = X_train.reshape(-1, 1, 16, 16)
    X_te = X_test.reshape(-1, 1, 16, 16)
    model = CNN2D()
    res = _train_loop(model, X_tr, y_train, X_te, y_test, epochs, batch_size, verbose=verbose)
    print(f'[CNN-Torch] Test acc {res["accuracy"]:.4f}, time {res["time"]:.2f}s '
          f'(device={get_device()})')
    return res


def train_cnn1d(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                kernel_size=5, pool_size=4, class_weights=None, use_focal=False, num_classes=NUM_CLASSES):
    # (N, D) -> (N, 1, D)
    X_tr = X_train.reshape(-1, 1, X_train.shape[1])
    X_te = X_test.reshape(-1, 1, X_test.shape[1])
    model = CNN1D(input_dim=X_train.shape[1], kernel_size=kernel_size,
                  pool_size=pool_size, num_classes=num_classes)
    res = _train_loop(model, X_tr, y_train, X_te, y_test, epochs, batch_size, verbose=verbose,
                      class_weights=class_weights, use_focal=use_focal)
    print(f'[CNN1D-Torch] Test acc {res["accuracy"]:.4f}, time {res["time"]:.2f}s '
          f'(device={get_device()})')
    return res


def train_cnn1d_dilated(X_train, y_train, X_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
                       class_weights=None, use_focal=False, num_classes=NUM_CLASSES):
    X_tr = X_train.reshape(-1, 1, X_train.shape[1])
    X_te = X_test.reshape(-1, 1, X_test.shape[1])
    model = CNN1DDilated(input_dim=X_train.shape[1], num_classes=num_classes)
    res = _train_loop(model, X_tr, y_train, X_te, y_test, epochs, batch_size, verbose=verbose,
                      class_weights=class_weights, use_focal=use_focal)
    print(f'[CNN1D-Dilated-Torch] Test acc {res["accuracy"]:.4f}, time {res["time"]:.2f}s '
          f'(device={get_device()})')
    return res


def predict(model, X, batch_size=512):
    """Generic prediction routine that supports CNN/CNN1D/Dilated."""
    device = next(model.parameters()).device
    model.eval()

    # Decide reshape based on model
    if isinstance(model, CNN2D):
        X_t = X.reshape(-1, 1, 16, 16)
    else:
        X_t = X.reshape(-1, 1, X.shape[1])

    X_t = torch.as_tensor(X_t, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i:i+batch_size].to(device, non_blocking=True)
            preds.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(preds)
