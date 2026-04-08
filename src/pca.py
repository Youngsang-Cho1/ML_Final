import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

def main():
    # Load data
    df = pd.read_csv('spotify_songs.csv')
    
    # 1. Clean the data
    # Select numerical columns suitable for PCA
    numeric_features = [
        'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms'
    ]
    
    # Extract numerical data
    X = df[numeric_features].copy()

    print(X.head())
    
    # Handle missing values by dropping rows with NaN
    X = X.dropna()

    print(f"Data shape after dropping missing values: {X.shape}")
    
    # 2. Standardize the data
    # PCA is sensitive to the scale of features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. Apply PCA
    pca = PCA()
    pca.fit(X_scaled)
    
    # Analyze variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = explained_variance.cumsum()
    
    # Commented code used to single out variance explained by each principal component
    '''
    print("\nExplained Variance Ratio by Component:")
    for i, exp_var in enumerate(explained_variance):
        print(f"PC{i+1}: {exp_var:.4f}")
    '''

    print("\nCumulative Explained Variance:")
    for i, cum_var in enumerate(cumulative_variance):
        print(f"Up to PC{i+1}: {cum_var:.4f}")
    

    # Commented code used to analyze composition within PC1 and PC2. 
    '''
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=numeric_features
    )
    
    
    print("\nTop absolute feature loadings for PC1:")
    print(loadings['PC1'].abs().sort_values(ascending=False).head(5))
    
    print("\nTop absolute feature loadings for PC2:")
    print(loadings['PC2'].abs().sort_values(ascending=False).head(5))
    '''

    # Plot Cumulative Explained Variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance Threshold')
    plt.axhline(y=0.9, color='g', linestyle='--', label='90% Variance Threshold')
    plt.title('PCA - Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pca_cumulative_variance.png')
    print("\nPlot saved successfully\n")


    # Using PC_1 to PC_7 successfuly represents over 70% of total variance 
    # and reduces dimentionality down from 13 to 7

    # change n_components to change number of principal components
    pca_7 = PCA(n_components=7)
    X_pca_7 = pca_7.fit_transform(X_scaled)
    
    print(f"Reduced from {X.shape[1]} to {X_pca_7.shape[1]} principal components")



if __name__ == "__main__":
    main()
