import pandas as pd

df = pd.read_parquet('data/dataset/clustered_songs.parquet')
duplicates = df.duplicated(subset=['track_name', 'track_artist'], keep=False)
num_duplicates = duplicates.sum()
print(f"Total rows in df: {len(df)}")
print(f"Number of rows that are duplicates based on name/artist: {num_duplicates}")

sample = df[duplicates][['track_name', 'track_artist', 'playlist_genre']].sort_values('track_name').head(10)
print("\nSample of duplicates:")
print(sample)
