import pandas as pd
import numpy as np
import os
from audio_utils import get_safe_name

def main():
    """
    Synthesizes all processed data into a single 'Source of Truth' Parquet file.
    Joins: Clustered Data + Original Spotify Metadata + Raw MFCC Embeddings.
    """
    # File paths for components
    original_csv = 'data/dataset/spotify_songs.csv'
    embedded_parquet = 'data/embeddings/audio_features.parquet'
    clustered_parquet = 'data/dataset/clustered_songs.parquet'
    output_master = 'data/dataset/master_music_data.parquet'

    print("Building Master Dataset...")

    # Load All Component Dataframes
    if not os.path.exists(original_csv):
        print(f"Error: Missing {original_csv}")
        return
    df_orig = pd.read_csv(original_csv)
    
    if not os.path.exists(clustered_parquet):
        print(f"Error: Missing {clustered_parquet}. Run pipeline first.")
        return
    df_clustered = pd.read_parquet(clustered_parquet)

    if not os.path.exists(embedded_parquet):
        print(f"Error: Missing {embedded_parquet}")
        return
    df_emb = pd.read_parquet(embedded_parquet)

    # Pre-processing: Generate standardized join keys
    print("Synthesizing join keys...")
    df_orig['safe_name'] = df_orig.apply(lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1)
    df_clustered['safe_name'] = df_clustered.apply(lambda r: get_safe_name(r['track_name'], r['track_artist']), axis=1)
    df_emb['safe_name'] = df_emb['track_id'] 

    # Perform SQL-style Inner Joins to ensure only complete records are kept
    print("Performing master join...")
    
    # 1. Merge Clustered Results with Original Metadata
    master_df = pd.merge(
        df_clustered, 
        df_orig.drop(columns=['track_name', 'track_artist', 'playlist_genre']), 
        on='safe_name', 
        how='inner'
    )
    
    # 2. Merge with Raw High-Dimensional Embeddings
    master_df = pd.merge(
        master_df,
        df_emb.drop(columns=['track_id']),
        on='safe_name', 
        how='inner'
    )

    # Final Cleanup: Remove duplicates and reset index
    master_df = master_df.drop_duplicates(subset=['safe_name']).reset_index(drop=True)
    
    print(f"Master Dataset created with {master_df.shape[0]} songs and {master_df.shape[1]} columns.")
    
    # Save as Parquet for fast I/O in the Streamlit app
    master_df.to_parquet(output_master)
    print(f"Successfully saved Master Dataset to {output_master}")

if __name__ == "__main__":
    main()
