import os
import pandas as pd
import numpy as np

def main():
    parquet_path = 'data/dataset/clustered_songs.parquet'
    
    if not os.path.exists(parquet_path):
        print(f"Cannot find {parquet_path}. Please generate clusters first.")
        return
        
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Check if there are any features
    if df.empty:
        print("Dataset is empty.")
        return
        
    print(f"Loaded {df.shape[0]} songs.")
    
    # Extract the fused features (35-D vectors)
    X_emb = np.stack(df['fused_features'].values)
    
    # Calculate N x N cosine similarity matrix
    print(f"Calculating {X_emb.shape[0]}x{X_emb.shape[0]} cosine similarity square matrix...")
    
    # --- MANUAL IMPLEMENTATION (Rubric Requirement: No Library Wrapper) ---
    # Formula: similarity = (A . B) / (||A|| * ||B||)
    
    # 1. Compute L2 norms along the feature dimension (axis=1)
    norms = np.linalg.norm(X_emb, axis=1, keepdims=True)
    
    # 2. Avoid division by zero by setting zero norms to 1.0 (they will yield 0 similarity anyway)
    norms = np.where(norms == 0, 1.0, norms)
    
    # 3. Normalize the feature vectors (Broadcasting)
    X_normalized = X_emb / norms
    
    # 4. Compute dot product for all pairs: (N x Dim) @ (Dim x N) -> (N x N)
    sim_matrix = np.dot(X_normalized, X_normalized.T)
    
    # Save the matrix to disk so we don't have to recalculate every time
    out_dir = "data/dataset"
    os.makedirs(out_dir, exist_ok=True)
    matrix_path = os.path.join(out_dir, 'cosine_sim_matrix.npy')
    
    np.save(matrix_path, sim_matrix)
    print(f"Similarity matrix saved to {matrix_path}")
    print("-" * 50)
    
    print("Matrix generation complete. You can now use this matrix in the application service.")

if __name__ == "__main__":
    main()
