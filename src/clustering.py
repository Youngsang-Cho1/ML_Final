import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # 1. Load Fused Features
    parquet_path = 'data/dataset/fused_features.parquet'
    if not os.path.exists(parquet_path):
        print(f"Cannot find {parquet_path}. Please run pca.py first.")
        return
        
    df = pd.read_parquet(parquet_path)
    print(f"Loaded fused features: {df.shape[0]} songs.")
    
    # Extract the fused feature matrix
    X_fused = np.stack(df['fused_features'].values)
    
    # 2. Final Normalization
    # Scale the newly fused 35D space to ensure Euclidean distance is stable
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_fused)
    
    # 3. K-Means Clustering
    n_clusters = 10  # Reduced clusters for ~1000 items
    print(f"Running K-Means with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_final)
    
    # 4. Save Results
    df['cluster'] = labels
    
    # Save the clustered data so that the recommendation system can use it for cosine similarity
    out_path = 'data/dataset/clustered_songs.parquet'
    df.to_parquet(out_path)
    print(f"Successfully clustered songs and saved to {out_path}")
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
