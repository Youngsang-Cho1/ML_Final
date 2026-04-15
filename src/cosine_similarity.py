import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
    sim_matrix = cosine_similarity(X_emb)
    
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
