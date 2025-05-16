import numpy as np
from sklearn.neighbors import NearestNeighbors

def frssa_borderline_sampler(X, y, minority_class=1, n_samples=1000, k_neighbors=5, expansion_weight=0.5, random_state=42):

    np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)

    # Separate classes
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]
    X_min = X[minority_idx]
    X_maj = X[majority_idx]

    n_min, n_maj = len(X_min), len(X_maj)
    n_needed = n_maj - n_min
    if n_needed <= 0:
        return np.empty((0, X.shape[1])), np.array([])

    # Step 1: KNN on entire set to detect borderline
    nn_global = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    _, indices = nn_global.kneighbors(X_min)

    hardness = np.array([
        np.sum(y[indices[i]] != minority_class) / k_neighbors
        for i in range(len(X_min))
    ])

    # Borderline = minority with 0 < hardness < 1
    borderline_mask = (hardness > 0) & (hardness < 1)
    X_border = X_min[borderline_mask]
    hardness = hardness[borderline_mask]

    # Step 2: Allocate synthetic samples based on hardness
    hardness_sum = np.sum(hardness)
    if hardness_sum == 0:
        hardness = np.ones_like(hardness) / len(hardness)
    else:
        hardness = hardness / hardness_sum

    sample_allocation = np.round(hardness * n_needed).astype(int)
    sample_allocation[np.argmax(hardness)] += (n_needed - sample_allocation.sum())

    # Step 3: Generate synthetic samples
    nn_border = NearestNeighbors(n_neighbors=k_neighbors).fit(X_border)
    _, border_neighbors = nn_border.kneighbors(X_border)

    synthetic_samples = []
    for i, n_gen in enumerate(sample_allocation):
        for _ in range(n_gen):
            j = np.random.choice(border_neighbors[i])
            x_i = X_border[i]
            x_j = X_border[j]
            alpha = np.random.rand()
            base = x_i + alpha * (x_j - x_i)

            # Apply expansion (random dispersion) based on ASST
            dispersion = np.random.normal(0, 0.1 * expansion_weight, size=X.shape[1])
            x_new = base + dispersion
            x_new = np.clip(x_new, X.min(axis=0), X.max(axis=0))
            synthetic_samples.append(x_new)

    X_syn = np.array(synthetic_samples)
    y_syn = np.full(X_syn.shape[0], minority_class)
    return X_syn, y_syn
