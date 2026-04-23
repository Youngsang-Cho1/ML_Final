"""
Manual PCA implementation using covariance matrix + SVD.
Used for 2D visualization of audio embeddings in the Streamlit Evaluation tab.
No sklearn dependency.
"""
import numpy as np


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize X to zero mean and unit variance (z-score normalization).
    This is critical before PCA: without it, high-variance features
    (e.g. tempo in BPM) would dominate the principal components purely
    because of their scale, not because they carry more information.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # avoid division by zero for constant features
    return (X - mean) / std, mean, std


def manual_pca(X: np.ndarray, n_components: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual PCA via covariance matrix eigendecomposition using SVD.

    Steps:
    1. Center X (subtract mean) so the covariance is computed around the origin.
    2. Compute the covariance matrix: C = X^T @ X / (n-1).
       Each entry C[i,j] measures how much feature i and j vary together.
    3. Apply SVD to C: C = U S V^T.
       - U: eigenvectors (principal component directions)
       - S: eigenvalues (variance explained per component)
       Since C is symmetric positive semi-definite, SVD and eigendecomposition
       are equivalent here. Columns of U are the principal axes.
    4. Project X onto the top-k principal components: X_pca = X_centered @ U[:, :k]

    Returns (X_pca, explained_variance_ratio).
    """
    n = X.shape[0]

    # Step 1: Center the data — PCA requires zero-mean input
    X_centered = X - X.mean(axis=0)

    # Step 2: Covariance matrix (n-1 denominator = unbiased estimate)
    cov = X_centered.T @ X_centered / (n - 1)

    # Step 3: SVD of covariance matrix
    U, S, _ = np.linalg.svd(cov)

    # Step 4: Project data onto the top-k principal components
    components = U[:, :n_components]
    X_pca = X_centered @ components

    explained_variance_ratio = S[:n_components] / S.sum()

    return X_pca, explained_variance_ratio
