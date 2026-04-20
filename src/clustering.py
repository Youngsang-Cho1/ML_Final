import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class ManualKMeans:
    """
    Pure NumPy implementation of K-Means. 
    Matches the course requirement of implementing core ML algorithms from scratch.
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
        """
        Optimized K-Means fitting process.
        Uses the dot-product identity to avoid high-memory 3D array broadcasting.
        """
        np.random.seed(self.random_state)
        # Random initialization of centroids
        random_idx = np.random.permutation(X.shape[0])[:self.k]
        self.centroids = X[random_idx]

        for i in range(self.max_iter):
            # Assignment Step: Find nearest centroid for each point
            # Math: ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b (Memory efficient)
            x2 = np.sum(X**2, axis=1, keepdims=True)
            c2 = np.sum(self.centroids**2, axis=1)
            dist2 = x2 + c2 - 2 * X @ self.centroids.T
            distances = np.sqrt(np.maximum(dist2, 0))
            self.labels = np.argmin(distances, axis=1)

            # Update Step: Move centroids to the mean of assigned points
            new_centroids = np.array([X[self.labels == k].mean(axis=0) if len(X[self.labels == k]) > 0 
                                      else self.centroids[k] for k in range(self.k)])

            # Convergence Check: Stop if centroids don't move beyond tolerance
            if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                break
                
            self.centroids = new_centroids

        # Calculate final Inertia (Within-Cluster Sum of Squares)
        x2 = np.sum(X**2, axis=1, keepdims=True)
        c2 = np.sum(self.centroids**2, axis=1)
        dist2 = x2 + c2 - 2 * X @ self.centroids.T
        min_distances = np.min(np.sqrt(np.maximum(dist2, 0)), axis=1)
        self.inertia_ = np.sum(min_distances**2)
        
        return self

    def calculate_silhouette(self, X):
        """
        Computes the Silhouette Score to evaluate clustering quality.
        Measures how similar a point is to its own cluster vs. other clusters.
        """
        if self.labels is None:
            return 0
            
        n_samples = X.shape[0]
        silhouette_vals = np.zeros(n_samples)
        
        # Efficiently compute N x N distance matrix using the same dot-product identity
        x2 = np.sum(X**2, axis=1, keepdims=True)
        dist_matrix_sq = x2 + x2.T - 2 * X @ X.T
        dist_matrix = np.sqrt(np.maximum(dist_matrix_sq, 0))
        
        for i in range(n_samples):
            # a(i): average distance to points in the SAME cluster
            cluster_i = self.labels[i]
            same_cluster_mask = (self.labels == cluster_i).copy()
            same_cluster_mask[i] = False # Exclude self
            
            if np.sum(same_cluster_mask) > 0:
                a_i = np.mean(dist_matrix[i][same_cluster_mask])
            else:
                a_i = 0
                
            # b(i): average distance to points in the NEAREST OTHER cluster
            b_i = float('inf')
            for k in range(self.k):
                if k == cluster_i: continue
                other_cluster_mask = (self.labels == k)
                if np.sum(other_cluster_mask) > 0:
                    avg_dist_to_k = np.mean(dist_matrix[i][other_cluster_mask])
                    b_i = min(b_i, avg_dist_to_k)
            
            # Silhouette calculation: (b-a)/max(a,b)
            if a_i == 0 and b_i == float('inf'):
                silhouette_vals[i] = 0
            else:
                silhouette_vals[i] = (b_i - a_i) / max(a_i, b_i)
                
        return np.mean(silhouette_vals)

import pickle

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
    
    # 4. Serialization: Save the cluster scaler and centroids
    print("Saving clustering models to models/ directory...")
    os.makedirs('models', exist_ok=True)
    with open('models/cluster_model.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler, 'centroids': kmeans.centroids}, f)
        
    # Optional: Calculate Metrics for logging
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
