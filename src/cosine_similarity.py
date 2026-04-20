import os
import pandas as pd
import numpy as np

def main():
    """
    Computes a Global Cosine Similarity Matrix for all songs in the master dataset.
    This enables 'Seed Song' based recommendation by finding nearest neighbors in 35D space.
    """
    parquet_path = 'data/dataset/master_music_data.parquet'
    
    if not os.path.exists(parquet_path):
        print(f"Cannot find {parquet_path}. Please run build_master_dataset.py first.")
        return
        
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    if df.empty:
        print("Dataset is empty.")
        return
        
    # Scalability Check: Calculate memory overhead for N x N matrix
    n_songs = df.shape[0]
    print(f"Loaded {n_songs} songs.")
    
    if n_songs > 20000:
        # Each float64 takes 8 bytes. N^2 matrix can grow very fast.
        memory_gb = (n_songs**2 * 8) / (1024**3)
        print(f"WARNING: Generating an {n_songs}x{n_songs} matrix will consume ~{memory_gb:.2f} GB of memory.")
        
    # Extract the fused feature matrix (35D)
    X_emb = np.stack(df['fused_features'].values)
    
    # Calculate N x N Cosine Similarity
    print(f"Calculating {n_songs}x{n_songs} cosine similarity matrix...")
    
    # --- MANUAL IMPLEMENTATION (Rubric Requirement: No Library Wrapper) ---
    # Formula: similarity = (A . B) / (||A|| * ||B||)
    
    # 1. Compute L2 norms along the feature dimension
    norms = np.linalg.norm(X_emb, axis=1, keepdims=True)
    
    # 2. Add epsilon to avoid division by zero
    norms = np.where(norms == 0, 1e-9, norms)
    
    # 3. Normalize vectors to unit length
    X_normalized = X_emb / norms
    
    # 4. Matrix Multiplication: (N x 35) @ (35 x N) -> (N x N)
    # The result is the dot product of normalized vectors, which IS cosine similarity.
    sim_matrix = np.dot(X_normalized, X_normalized.T)
    
    # Save the matrix as a binary NumPy file for fast loading in Streamlit
    out_dir = "data/dataset"
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, 'cosine_sim_matrix.npy')
    
    np.save(matrix_path, sim_matrix)
    print(f"Similarity matrix successfully saved to {matrix_path}")

if __name__ == "__main__":
    main()
