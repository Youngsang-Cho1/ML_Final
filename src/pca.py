import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.audio_utils import get_safe_name
import pickle

def main():
    """
    Multimodal Fusion Pipeline:
    1. Loads Spotify metadata (tabular) and Audio features (embeddings).
    2. Uses PCA for dimensionality reduction to keep the most significant variance.
    3. Concatenates them into a unified 35D vector space (The 'Fused' space).
    4. Saves models for real-time 'New Song' analysis.
    """
    # 1. Load Spotify Metadata (Tabular features like energy, danceability)
    csv_path = 'data/dataset/spotify_songs.csv'
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Standardize song names for joining across different data sources
    df['safe_name'] = df.apply(lambda row: get_safe_name(row['track_name'], row['track_artist']), axis=1)
    
    # Avoid duplicate data inflation
    df = df.drop_duplicates(subset=['safe_name']).reset_index(drop=True)
    
    # 2. Load Audio Embeddings (MFCC/Spectral features)
    parquet_path = 'data/embeddings/audio_features.parquet'
    if not os.path.exists(parquet_path):
        print(f"Cannot find {parquet_path}. Run audio_feature_extractor.py first.")
        return
        
    embeddings_df = pd.read_parquet(parquet_path)
        
    # Join Metadata and Audio Embeddings
    merged_df = pd.merge(df, embeddings_df, left_on='safe_name', right_on='track_id', how='inner')
    
    if merged_df.empty:
        print("No matching records found after merge.")
        return

    # 3. Dimensionality Reduction on Spotify Features (13D -> 11D for ~92% variance)
    numeric_features = [
        'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms'
    ]
    X_num = merged_df[numeric_features].copy().dropna()
    merged_df = merged_df.loc[X_num.index].reset_index(drop=True)
    
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)
    pca_num = PCA(n_components=11)
    X_num_pca = pca_num.fit_transform(X_num_scaled)
    
    # 4. Dimensionality Reduction on Audio Embeddings (58D -> 34D for 95% variance)
    X_emb = np.stack(merged_df['embedding'].values)
    scaler_emb = StandardScaler()
    X_emb_scaled = scaler_emb.fit_transform(X_emb)
    
    pca_emb = PCA(n_components=34)
    X_emb_pca = pca_emb.fit_transform(X_emb_scaled)
    
    # 5. Multimodal Fusion (Concatenation)
    X_fused = np.hstack((X_num_pca, X_emb_pca))
    
    # 6. Serialization: Save models for real-time inference
    print("Saving PCA models and Scalers to models/ directory...")
    os.makedirs('models', exist_ok=True)
    with open('models/metadata_pca.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler_num, 'pca': pca_num}, f)
    with open('models/audio_pca.pkl', 'wb') as f:
        pickle.dump({'scaler': scaler_emb, 'pca': pca_emb}, f)
    
    # Save the fused features for clustering and retrieval
    out_df = merged_df[['track_name', 'track_artist', 'playlist_genre']].copy()
    out_df['spotify_id'] = merged_df['track_id_x'] if 'track_id_x' in merged_df.columns else merged_df['track_id']
    out_df['fused_features'] = list(X_fused)
    
    out_path = 'data/dataset/fused_features.parquet'
    out_df.to_parquet(out_path)
    print(f"Successfully saved 35D fused multimodal features to {out_path}")

if __name__ == "__main__":
    main()
