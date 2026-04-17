import numpy as np

def compute_target_similarities(target_vector: np.ndarray, dataset_matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between a target song vector and an entire dataset matrix.
    Operates strictly in O(N) memory complexity, making it safe for massive datasets.
    """
    # 1. Normalize the target vector
    target_norm = np.linalg.norm(target_vector)
    if target_norm == 0:
        target_norm = 1.0
    target_normalized = target_vector / target_norm
    
    # 2. Normalize the dataset matrix
    matrix_norms = np.linalg.norm(dataset_matrix, axis=1, keepdims=True)
    matrix_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    dataset_normalized = dataset_matrix / matrix_norms
    
    # 3. Compute dot product (1 x Dim) @ (Dim x N) -> (1 x N)
    sim_scores = np.dot(dataset_normalized, target_normalized)
    
    return sim_scores
