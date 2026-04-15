import pandas as pd
import os
import shutil

def main():
    original_csv_path = 'data/dataset/spotify_songs.csv'
    backup_csv_path = 'data/dataset/spotify_songs_full.csv'
    
    if not os.path.exists(original_csv_path):
        print(f"Error: {original_csv_path} not found.")
        if os.path.exists(backup_csv_path):
            print("Backup found, restoring from backup...")
            shutil.copy(backup_csv_path, original_csv_path)
        else:
            return

    # Backup the original 30k dataset if not already backed up
    if not os.path.exists(backup_csv_path):
        print("Backing up original full dataset...")
        shutil.copy(original_csv_path, backup_csv_path)
    
    # Load the full dataset (from backup to ensure we always sample from the 30k pool)
    df = pd.read_csv(backup_csv_path)
    print(f"Loaded full dataset: {len(df)} songs")
    
    # Count how many genres there are (Usually 6: pop, rap, rock, latin, r&b, edm)
    genres = df['playlist_genre'].unique()
    num_genres = len(genres)
    
    # Target 1000 songs -> Calculate how many per genre
    target_total = 1000
    samples_per_genre = target_total // num_genres
    print(f"Sampling {samples_per_genre} songs per genre ({num_genres} genres)...")
    
    # Perform Stratified Random Sampling
    # group by genre and sample 'samples_per_genre' per group, fixing random state for reproducibility
    sampled_df = df.groupby('playlist_genre').sample(n=samples_per_genre, random_state=42)
    
    # Shuffle the final dataframe so genres are mixed (otherwise it's sorted by genre chunks)
    sampled_df = sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Overwrite the main csv that the pipeline uses
    sampled_df.to_csv(original_csv_path, index=False)
    
    print("\n--- Sampling Complete ---")
    print(f"Created balanced dataset with {len(sampled_df)} songs and saved to {original_csv_path}.")
    print("Distribution:")
    print(sampled_df['playlist_genre'].value_counts())
    print("\nNext Steps:")
    print("1. Clear out your old data/audio and data/embeddings directories.")
    print("2. Run your fetch/generate pipeline from the beginning.")

if __name__ == "__main__":
    main()
