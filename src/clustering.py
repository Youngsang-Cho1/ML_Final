import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Default numerical Spotify audio features used for clustering
DEFAULT_FEATURES = [
    "danceability",
    "energy",
    "key",
    "loudness",
    "mode",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
]


def zscore_standardize(X: np.ndarray):
    """
    Standardize each column to mean 0 and std 1 without sklearn.
    Returns:
        X_scaled, mean, std
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def choose_feature_columns(df: pd.DataFrame, user_features=None):
    """
    Choose numeric audio feature columns.
    """
    if user_features:
        features = [col.strip() for col in user_features.split(",")]
    else:
        features = [col for col in DEFAULT_FEATURES if col in df.columns]

    if not features:
        raise ValueError("No valid feature columns were found in the dataset.")

    return features


class KMeans:
    """
    Manual K-Means implementation without sklearn.

    Parameters
    ----------
    n_clusters : int
        Number of clusters.
    max_iter : int
        Maximum number of iterations per run.
    tol : float
        Convergence tolerance.
    n_init : int
        Number of random restarts.
    random_state : int
        Random seed.
    standardize : bool
        Whether to z-score standardize features inside the class.
    """

    def __init__(
        self,
        n_clusters: int = 5,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: int = 42,
        standardize: bool = True,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state
        self.standardize = standardize

        self.labels_ = None
        self.centroids_ = None
        self.inertia_ = None

        self.mean_ = None
        self.std_ = None
        self.n_features_in_ = None

    def _validate_input(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        if X.shape[0] == 0:
            raise ValueError("X must contain at least one sample.")

        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")

        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot be greater than number of samples.")

        return X

    def _kmeans_plus_plus_init(self, X: np.ndarray, rng: np.random.Generator):
        """
        K-means++ initialization without sklearn.
        """
        n_samples = X.shape[0]
        centroids = np.empty((self.n_clusters, X.shape[1]), dtype=float)

        # First centroid
        first_idx = rng.integers(0, n_samples)
        centroids[0] = X[first_idx]

        # Remaining centroids
        for i in range(1, self.n_clusters):
            distances_sq = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[:i][np.newaxis, :, :]) ** 2, axis=2),
                axis=1,
            )

            total = distances_sq.sum()

            if total == 0:
                random_idx = rng.integers(0, n_samples)
                centroids[i] = X[random_idx]
                continue

            probabilities = distances_sq / total
            next_idx = rng.choice(n_samples, p=probabilities)
            centroids[i] = X[next_idx]

        return centroids

    def _assign_clusters(self, X: np.ndarray, centroids: np.ndarray):
        """
        Assign each sample to the nearest centroid.
        """
        distances_sq = np.sum(
            (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
            axis=2,
        )
        return np.argmin(distances_sq, axis=1)

    def _update_centroids(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ):
        """
        Update centroid of each cluster.
        If a cluster becomes empty, reinitialize it with a random point.
        """
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features), dtype=float)

        for cluster_id in range(self.n_clusters):
            cluster_points = X[labels == cluster_id]

            if len(cluster_points) == 0:
                random_idx = rng.integers(0, X.shape[0])
                new_centroids[cluster_id] = X[random_idx]
            else:
                new_centroids[cluster_id] = cluster_points.mean(axis=0)

        return new_centroids

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray):
        """
        Sum of squared distances from each point to its assigned centroid.
        """
        return np.sum((X - centroids[labels]) ** 2)

    def _prepare_fit_data(self, X: np.ndarray):
        X = self._validate_input(X)
        self.n_features_in_ = X.shape[1]

        if self.standardize:
            X_scaled, mean, std = zscore_standardize(X)
            self.mean_ = mean
            self.std_ = std
            return X_scaled

        self.mean_ = None
        self.std_ = None
        return X

    def _prepare_predict_data(self, X: np.ndarray):
        X = self._validate_input(X)

        if self.n_features_in_ is None:
            raise ValueError("Model has not been fitted yet.")

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but model was fitted with {self.n_features_in_}."
            )

        if self.standardize:
            X = (X - self.mean_) / self.std_

        return X

    def fit(self, X: np.ndarray):
        """
        Run full K-means with multiple random restarts.
        Keep the run with the lowest inertia.
        """
        X = self._prepare_fit_data(X)

        best_labels = None
        best_centroids = None
        best_inertia = np.inf

        base_rng = np.random.default_rng(self.random_state)

        for _ in range(self.n_init):
            run_seed = int(base_rng.integers(0, 10**9))
            rng = np.random.default_rng(run_seed)

            centroids = self._kmeans_plus_plus_init(X, rng)

            for _ in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels, rng)

                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids

                if shift < self.tol:
                    break

            labels = self._assign_clusters(X, centroids)
            inertia = self._compute_inertia(X, labels, centroids)

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()

        self.labels_ = best_labels
        self.centroids_ = best_centroids
        self.inertia_ = float(best_inertia)

        return self

    def predict(self, X: np.ndarray):
        """
        Assign new samples to the nearest fitted centroid.
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet.")

        X = self._prepare_predict_data(X)
        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X: np.ndarray):
        """
        Fit K-Means and return cluster labels.
        """
        self.fit(X)
        return self.labels_

    def get_centroids(self, original_scale: bool = True):
        """
        Return centroids.
        If standardize=True and original_scale=True, convert back to original feature scale.
        """
        if self.centroids_ is None:
            raise ValueError("Model has not been fitted yet.")

        if self.standardize and original_scale:
            return self.centroids_ * self.std_ + self.mean_

        return self.centroids_


def main():
    parser = argparse.ArgumentParser(description="Manual K-means clustering without sklearn.")
    parser.add_argument(
        "--input",
        type=str,
        default="data/dataset/spotify_songs.csv",
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset/spotify_songs_with_clusters.csv",
        help="Path to output CSV file with cluster labels.",
    )
    parser.add_argument(
        "--centroids_output",
        type=str,
        default="data/dataset/kmeans_centroids.csv",
        help="Path to output CSV file for centroid values.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of clusters.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Comma-separated feature columns. If omitted, default Spotify audio features are used.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=300,
        help="Maximum number of iterations.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance.",
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=10,
        help="Number of random restarts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    feature_cols = choose_feature_columns(df, args.features)

    # Keep only rows with complete feature data
    working_df = df.dropna(subset=feature_cols).copy()
    X = working_df[feature_cols].to_numpy(dtype=float)

    model = KMeans(
        n_clusters=args.k,
        max_iter=args.max_iter,
        tol=args.tol,
        n_init=args.n_init,
        random_state=args.seed,
        standardize=True,
    )

    labels = model.fit_predict(X)
    centroids_original = model.get_centroids(original_scale=True)

    working_df["cluster"] = labels
    working_df.to_csv(args.output, index=False)

    centroids_df = pd.DataFrame(centroids_original, columns=feature_cols)
    centroids_df.insert(0, "cluster", np.arange(args.k))
    centroids_df.to_csv(args.centroids_output, index=False)

    print("Done.")
    print(f"Rows clustered: {len(working_df)}")
    print(f"Features used: {feature_cols}")
    print(f"Number of clusters (k): {args.k}")
    print(f"Best inertia: {model.inertia_:.4f}")
    print(f"Clustered data saved to: {args.output}")
    print(f"Centroids saved to: {args.centroids_output}")
    print()
    print("Cluster sizes:")
    print(working_df["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()