import numpy as np
import pandas as pd

def manual_cosine_similarity(target_vector: np.ndarray, matrix: np.ndarray):
    """
    using Numpy to calculate the cosine similarity between a single target vector 
    and a matrix of other vectors
    """

    # 1. Calculate the dot product between the target song vector and every song vector in the matrix
    # since target_vector has shape (1, M), matrix has shape (N, M), matrix.T has shape (M, N)
    # np.dot(target_vector, matrix.T) results in a matrix of shape (1, N), producing one dot product score for each of the N songs
    # Then convert the result from shape (1, N) to a simpler 1D array of shape (N,) (using [0])
    dot_product = np.dot(target_vector, matrix.T)[0] 
    
    # 2. Calculate Magnitudes (||A|| and ||B||)
    # Compute the magnitude (Euclidean norm) of the target vector
    # It measures the length of the target song vector 
    norm_target = np.linalg.norm(target_vector)
    
    # This computes the norm along the feature dimension for each song in the matrix
    norms_matrix = np.linalg.norm(matrix, axis=1) 
    
    # Avoid division by zero by setting 0 norms to a tiny epsilon
    norms_matrix[norms_matrix == 0] = 1e-10
    if norm_target == 0:
        norm_target = 1e-10
        
    # 3. Compute Similarity Score
    sim_scores = dot_product / (norm_target * norms_matrix)
    
    return sim_scores

def rank_songs_in_cluster(target_idx: int, df: pd.DataFrame, features_matrix: np.ndarray, cluster_labels: np.ndarray, top_n: int = 5):
    """
    Filters the dataset to only include songs in the same cluster as the target song.
    Then, calculates cosine similarity manually between the target song and the songs in the same cluster.
    Returns the top_n most similar songs.
    """
    # 1. Identify which cluster the target song belongs to
    target_cluster = cluster_labels[target_idx]
    
    # 2. Find all indices of songs that belong to this same cluster
    cluster_indices = np.where(cluster_labels == target_cluster)[0]
    
    # 3. Extract the features only for the songs within this cluster
    cluster_features = features_matrix[cluster_indices]
    
    # 4. Extract the features for our target song
    target_features = features_matrix[target_idx].reshape(1, -1)
    
    # 5. Calculate cosine similarity manually
    sim_scores = manual_cosine_similarity(target_features, cluster_features)
    
    # 6. Pair the original DataFrame index with its similarity score
    sim_scores_indexed = list(zip(cluster_indices, sim_scores))
    
    # 7. Sort by similarity score in descending order (highest similarity first)
    sim_scores_indexed.sort(key=lambda x: x[1], reverse=True)
    
    # 8. Extract the top_n results (skipping the exact original target song)
    top_indices = []
    for original_idx, score in sim_scores_indexed:
        if original_idx != target_idx:
            top_indices.append(original_idx)
        if len(top_indices) == top_n:
            break
            
    # 9. Return the corresponding recommended rows from the original dataframe
    return df.iloc[top_indices]
