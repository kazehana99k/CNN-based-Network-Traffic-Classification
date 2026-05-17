"""
Feature engineering utilities:
1. Correlation-based feature reordering - cluster similar features together for 1D Conv
2. ATTACK-targeted feature engineering - emphasize port / flag features
3. Feature statistics inspection - help identify which columns are which type

Note: the Moore dataset's exact discriminator -> column mapping is not public,
but we identify port columns by data heuristics (first 2 columns are integer-valued and look like port numbers).
"""
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


# ============== Feature Reordering ==============

def correlation_reorder(X, sample_size=20000, seed=0):
    """Reorder feature columns based on absolute Pearson correlation.

    Idea: After hierarchical clustering using |corr| as similarity, the leaf
    order in the dendrogram places highly correlated features adjacent. This
    aligns with the goal of "features in the same semantic group sit together",
    benefiting 1D Conv to capture intra-group patterns.

    Returns:
        order: np.ndarray, shape (n_features,) - new column index order
    """
    n_features = X.shape[1]

    rng = np.random.default_rng(seed)
    if len(X) > sample_size:
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X_sub = X[idx]
    else:
        X_sub = X

    # Drop columns with all-zero / no-variance to avoid corrcoef warnings
    stds = X_sub.std(axis=0)
    valid_mask = stds > 1e-12

    if valid_mask.sum() < 2:
        return np.arange(n_features)

    X_valid = X_sub[:, valid_mask]
    valid_idx = np.where(valid_mask)[0]
    invalid_idx = np.where(~valid_mask)[0]

    # Compute correlation matrix
    corr = np.corrcoef(X_valid.T)
    corr = np.nan_to_num(corr, nan=0.0)

    # Convert to distance matrix in [0,2]
    dist = 1.0 - np.abs(corr)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # enforce symmetry

    # Hierarchical clustering, get leaf ordering
    Z = linkage(squareform(dist, checks=False), method='average')
    leaf_order = leaves_list(Z)

    # Concatenate: valid columns (reordered) + invalid columns (kept at the tail)
    new_order = np.concatenate([valid_idx[leaf_order], invalid_idx])
    return new_order


def apply_reorder(X, order):
    """Apply the column ordering."""
    return X[:, order]


# ============== ATTACK-Targeted Feature Engineering ==============

# Based on data inspection: the first 2 columns are clearly port numbers
# (integer-valued, range typically 0-65535, well-known ports are < 1024).
PORT_COL_SERVER = 0
PORT_COL_CLIENT = 1


def add_attack_features(X, port_server_col=PORT_COL_SERVER,
                       port_client_col=PORT_COL_CLIENT):
    """Add port-related derived features for ATTACK class.

    ATTACK packets often disguise themselves with well-known ports of other
    types (port 80 for HTTP, 443 for HTTPS) but the real port distribution
    differs. We add:
    1. Well-known port indicator (port < 1024)
    2. Ephemeral port indicator (port >= 49152)
    3. log1p(port) - reduce extreme port-value skew
    4. Port difference / ratio - same vs different ports usually correlate with traffic type

    Args:
        X: shape (N, D), port columns must NOT yet be normalized (i.e., raw integer-valued)
    Returns:
        X_new: shape (N, D+6) - 6 additional features appended
    """
    n_samples = X.shape[0]

    # Note: if X has gone through L1 normalization, the port values won't be raw integers anymore.
    # We need to use raw integer values, so this must be called BEFORE normalize_features.
    ps = X[:, port_server_col].astype(np.float64)
    pc = X[:, port_client_col].astype(np.float64)

    # 1. Well-known port indicator
    is_well_known_s = (ps < 1024).astype(np.float64)
    is_well_known_c = (pc < 1024).astype(np.float64)

    # 2. Ephemeral port indicator
    is_ephemeral_s = (ps >= 49152).astype(np.float64)
    is_ephemeral_c = (pc >= 49152).astype(np.float64)

    # 3. log1p transform of port (reduce extreme skew, exposes mid-range patterns to the network)
    log_ps = np.log1p(np.abs(ps))
    log_pc = np.log1p(np.abs(pc))

    new_feats = np.column_stack([
        is_well_known_s, is_well_known_c,
        is_ephemeral_s, is_ephemeral_c,
        log_ps, log_pc,
    ])

    X_new = np.hstack([X, new_feats])
    return X_new


# ============== Feature Statistics Inspection ==============

def feature_stats(X, top_k=10):
    """Print basic statistics for inspection."""
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    maxs = X.max(axis=0)
    mins = X.min(axis=0)

    print(f'Feature dims: {X.shape[1]}')
    print(f'Top {top_k} columns:')
    for i in range(min(top_k, X.shape[1])):
        print(f'  Column {i:3d}: min={mins[i]:10.3f}, max={maxs[i]:10.3f}, '
              f'mean={means[i]:10.3f}, std={stds[i]:10.3f}')
