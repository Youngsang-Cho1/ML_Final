import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def main():
    # 1. Load Spotify Metadata
    csv_path = 'data/dataset/spotify_songs.csv'
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Create the exact same 'safe_name' string to accurately join with the embedding parquet
    # Since fetch_youtube_audio.py saves the files using safe_name, the embed_dir gets that name as track_id
    df['safe_name'] = df.apply(lambda row: "".join([c for c in f"{row['track_name']} - {row['track_artist']}" if c.isalpha() or c.isdigit() or c==' ']).rstrip(), axis=1)
    
    # 2. Load Embeddings
    parquet_path = 'data/embeddings/embedded_spectrograms.parquet'
    if not os.path.exists(parquet_path):
        print(f"Cannot find {parquet_path}. Have you run fetch_youtube_audio -> generate_spectrograms -> spectrogram_embedding?")
        return
        
    try:
        embeddings_df = pd.read_parquet(parquet_path)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
        
    # Merge on the safe_name
    print("Merging Spotify dataset with DINOv2 audio embeddings...")
    merged_df = pd.merge(df, embeddings_df, left_on='safe_name', right_on='track_id', how='inner')
    print(f"Merged Data Shape: {merged_df.shape}")
    
    if merged_df.empty:
        print("No matching records found after merge.")
        return

    # 3. Process Original Numeric Features (13 features)
    numeric_features = [
        'track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence',
        'tempo', 'duration_ms'
    ]
    
    X_num = merged_df[numeric_features].copy()
    
    # Handle missing values
    valid_indices = X_num.dropna().index
    X_num = X_num.loc[valid_indices]
    merged_df = merged_df.loc[valid_indices].reset_index(drop=True)
    
    # Standardize numerical features
    scaler_num = StandardScaler()
    X_num_scaled = scaler_num.fit_transform(X_num)
    
    # PCA on Numerical Features (Dimensionality: 13 -> 5)
    n_comp_num = min(5, X_num_scaled.shape[0])
    pca_num = PCA(n_components=n_comp_num)
    X_num_pca = pca_num.fit_transform(X_num_scaled)
    print(f"Spotify Features reduced via PCA: {X_num_scaled.shape[1]} -> {X_num_pca.shape[1]} dimensions")
    
    # 4. Process DINOv2 Embeddings
    X_emb = np.stack(merged_df['embedding'].values)
    
    # Standardize embeddings
    scaler_emb = StandardScaler()
    X_emb_scaled = scaler_emb.fit_transform(X_emb)
    
    # PCA on Embeddings (Dimensionality: 768 -> 30)
    n_comp_emb = min(30, X_emb_scaled.shape[0], X_emb_scaled.shape[1])
    pca_emb = PCA(n_components=n_comp_emb)
    X_emb_pca = pca_emb.fit_transform(X_emb_scaled)
    print(f"Audio Embeddings reduced via PCA: {X_emb_scaled.shape[1]} -> {X_emb_pca.shape[1]} dimensions")
    
    # 5. Multimodal Fusion (Concatenation)
    X_fused = np.hstack((X_num_pca, X_emb_pca))
    print(f"Final Fused Data Shape: {X_fused.shape[1]} dimensions (Ready for Clustering)")
    
    # Save the fused features and track metadata so clustering.py can use it
    out_df = merged_df[['track_name', 'track_artist', 'playlist_genre']].copy()
    
    # Rename for safety
    if 'track_id_x' in merged_df.columns:
        out_df['spotify_id'] = merged_df['track_id_x']
    elif 'track_id' in merged_df.columns:
        out_df['spotify_id'] = merged_df['track_id']
        
    out_df['fused_features'] = list(X_fused)
    
    out_path = 'data/dataset/fused_features.parquet'
    out_df.to_parquet(out_path)
    print(f"Successfully saved fused multimodal features to {out_path}")

if __name__ == "__main__":
    main()
