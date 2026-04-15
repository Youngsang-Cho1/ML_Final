import pandas as pd
import numpy as np
import os

def main():
    # File paths
    original_csv = 'data/dataset/spotify_songs.csv'
    embedded_parquet = 'data/embeddings/embedded_spectrograms.parquet'
    clustered_parquet = 'data/dataset/clustered_songs.parquet'
    output_master = 'data/dataset/master_music_data.parquet'

    print("Building Master Dataset...")

    # 1. Load Original Data
    if not os.path.exists(original_csv):
        print(f"Error: Missing {original_csv}")
        return
    df_orig = pd.read_csv(original_csv)
    
    # 2. Load Clustered Data (Result of PCA + K-Means)
    if not os.path.exists(clustered_parquet):
        print(f"Error: Missing {clustered_parquet}. Run pipeline first.")
        return
    df_clustered = pd.read_parquet(clustered_parquet)

    # 3. Load Raw DINOv2 Embeddings (384D)
    if not os.path.exists(embedded_parquet):
        print(f"Error: Missing {embedded_parquet}")
        return
    df_emb = pd.read_parquet(embedded_parquet)

    # Pre-processing for clean join
    # Create safe names for joining across different scripts
    def get_safe_name(name, artist):
        return "".join([c for c in f"{name} - {artist}" if c.isalpha() or c.isdigit() or c==' ']).rstrip()

    print("Synthesizing join keys...")
    df_orig['safe_name'] = df_orig.apply(lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1)
    
    # df_orig already has 'safe_name' from the apply above
    # Create safe_name for others
    df_clustered['safe_name'] = df_clustered.apply(lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1)
    df_emb['safe_name'] = df_emb['track_id'] # track_id in embeddings is already the safe_name

    # 4. Perform SQL-style Joins
    print("Performing master join...")
    
    # Merge Clustered + Original Metadata/Features
    master_df = pd.merge(
        df_clustered, 
        df_orig.drop(columns=['track_name', 'track_artist', 'playlist_genre']), 
        on='safe_name', 
        how='inner'
    )
    
    # Merge with Raw Embeddings
    master_df = pd.merge(
        master_df,
        df_emb.drop(columns=['track_id']),
        on='safe_name', 
        how='inner'
    )

    # 5. Cleanup
    # drop duplicates if any arose from the metadata join
    master_df = master_df.drop_duplicates(subset=['safe_name']).reset_index(drop=True)
    
    # Identify key columns to keep for the final "Source of Truth"
    # Metadata + Original 13 + Fused 35 + Cluster + Full 384 Embedding
    print(f"Master Dataset created with {master_df.shape[0]} songs and {master_df.shape[1]} columns.")
    
    # 6. Save
    master_df.to_parquet(output_master)
    print(f"Successfully saved Master Dataset to {output_master}")

if __name__ == "__main__":
    main()
