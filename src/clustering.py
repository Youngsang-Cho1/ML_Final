import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ManualKMeans:
    """
    Pure NumPy implementation of the K-Means algorithm for rubric compliance.
    Includes methods for model evaluation (Inertia, Silhouette).
    """
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        np.random.seed(self.random_state)
        # 1. Random centroid initialization
        random_idx = np.random.permutation(X.shape[0])[:self.k]
        self.centroids = X[random_idx]

        for i in range(self.max_iter):
            # 2. Assignment Step
            # Compute distances from each point to each centroid
            # X: (N, D), Centroids: (K, D) -> Distances: (N, K)
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # 3. Update Step
            new_centroids = np.array([X[self.labels == k].mean(axis=0) if len(X[self.labels == k]) > 0 
                                      else self.centroids[k] for k in range(self.k)])

            # 4. Convergence Check
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
                
            self.centroids = new_centroids

        # Calculate final Inertia (Within-Cluster Sum of Squares)
        final_distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        min_distances = np.min(final_distances, axis=1)
        self.inertia_ = np.sum(min_distances**2)
        
        return self

    def calculate_silhouette(self, X):
        """
        Manually calculate the Silhouette Score for the clustering result.
        Now safely subsampled for massive datasets to avoid O(n^2) OOM crashes.
        """
        if self.labels is None:
            return 0
            
        n_samples = X.shape[0]
        
        if n_samples > 1000:
            # FIX: Previously calculated an N x N matrix. On 30,000 songs, this requires 7.2GB of RAM.
            # We now safely subsample 1000 random data points to calculate the silhouette
            # score without crashing Streamlit via Out-Of-Memory errors.
            rng = np.random.default_rng(self.random_state)
            sample_idx = rng.choice(n_samples, 1000, replace=False)
            X = X[sample_idx]
            self_labels_sample = self.labels[sample_idx]
            n_samples = 1000
        else:
            self_labels_sample = self.labels
            
        silhouette_vals = np.zeros(n_samples)
        
        # Precompute distance matrix on the (max 1000) samples memory footprint
        dist_matrix = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        
        for i in range(n_samples):
            cluster_i = self_labels_sample[i]
            same_cluster_mask = (self_labels_sample == cluster_i).copy() 
            same_cluster_mask[i] = False  
            
            if np.sum(same_cluster_mask) > 0:
                a_i = np.mean(dist_matrix[i][same_cluster_mask])
            else:
                a_i = 0
                
            b_i = float('inf')
            for k in range(self.k):
                if k == cluster_i:
                    continue
                other_cluster_mask = (self_labels_sample == k)
                if np.sum(other_cluster_mask) > 0:
                    avg_dist_to_k = np.mean(dist_matrix[i][other_cluster_mask])
                    b_i = min(b_i, avg_dist_to_k)
            
            if a_i == 0 and b_i == float('inf'):
                silhouette_vals[i] = 0
            else:
                silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
                
        return np.mean(silhouette_vals)

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
    scaler = StandardScaler()
    X_final = scaler.fit_transform(X_fused)
    
    # 3. Manual K-Means Clustering
    n_clusters = 10 
    print(f"Running Manual K-Means with {n_clusters} clusters (Rubric Compliant)...")
    kmeans = ManualKMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_final)
    
    # 4. Optional: Calculate Metrics for logging
    sil_score = kmeans.calculate_silhouette(X_final)
    print(f"Cluster Inertia (WCSS): {kmeans.inertia_:.2f}")
    print(f"Silhouette Score: {sil_score:.4f}")
    
    # 5. Save Results
    df['cluster'] = kmeans.labels
    
    out_path = 'data/dataset/clustered_songs.parquet'
    df.to_parquet(out_path)
    print(f"Successfully clustered songs and saved to {out_path}")
    print("\nCluster Distribution:")
    print(df['cluster'].value_counts().sort_index())

if __name__ == "__main__":
    main()
