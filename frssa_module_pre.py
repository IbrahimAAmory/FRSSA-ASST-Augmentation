import numpy as np
from sklearn.neighbors import NearestNeighbors

def frssa_borderline_sampler(X, y, minority_class=1, k_neighbors=5, random_state=42):
   
    np.random.seed(random_state)
    X = np.array(X)
    y = np.array(y)

    # Separate minority and majority
    minority_idx = np.where(y == minority_class)[0]
    majority_idx = np.where(y != minority_class)[0]
    X_min, X_maj = X[minority_idx], X[majority_idx]

    # Determine how many synthetic samples are needed
    n_min, n_maj = len(X_min), len(X_maj)
    n_needed = n_maj - n_min
    if n_needed <= 0:
        return X, y

    # KNN on full data to find borderline minority samples
    nn_full = NearestNeighbors(n_neighbors=k_neighbors).fit(X)
    _, full_neighbors = nn_full.kneighbors(X_min)

    # Count how many neighbors are majority for each minority sample (borderline detection)
    hardness = np.array([
        np.sum(y[full_neighbors[i]] != minority_class) / k_neighbors
        for i in range(n_min)
    ])
    borderline_mask = (hardness > 0) & (hardness < 1)
    X_border = X_min[borderline_mask]
    hardness = hardness[borderline_mask]

    # Normalize hardness to allocate samples
    hardness_sum = hardness.sum()
    hardness = hardness / hardness_sum if hardness_sum > 0 else np.ones_like(hardness) / len(hardness)
    sample_allocation = np.round(hardness * n_needed).astype(int)

    # Fix rounding mismatch
    diff = n_needed - sample_allocation.sum()
    if diff != 0:
        max_idx = np.argmax(sample_allocation)
        sample_allocation[max_idx] += diff

    # KNN within minority class
    nn_border = NearestNeighbors(n_neighbors=k_neighbors).fit(X_border)
    _, border_neighbors = nn_border.kneighbors(X_border)

    synthetic_samples = []

    for i, n_samples in enumerate(sample_allocation):
        for _ in range(n_samples):
            j = np.random.choice(border_neighbors[i])
            x_i = X_border[i]
            x_j = X_border[j]
            alpha = np.random.rand()
            x_new = x_i + alpha * (x_j - x_i)

            # Optional: clipping to feature range to prevent unrealistic values
            x_new = np.clip(x_new, X.min(axis=0), X.max(axis=0))

            synthetic_samples.append(x_new)

    X_syn = np.array(synthetic_samples)
    y_syn = np.full(X_syn.shape[0], minority_class)

    # Final augmented set
    X_aug = np.vstack((X, X_syn))
    y_aug = np.hstack((y, y_syn))
    return X_aug, y_aug
