import os
import time
import requests
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv('SPOTIPY_CLIENT_ID')
CLIENT_SECRET = os.getenv('SPOTIPY_CLIENT_SECRET')

def download_audio(url, path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        return True
    return False

def main():
    if not CLIENT_ID or not CLIENT_SECRET or 'your_' in CLIENT_ID:
        print("Please configure your SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in the .env file.")
        return

    print("Initializing Spotipy client...")
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    csv_path = "data/dataset/spotify_songs.csv"
    output_dir = "data/audio_files"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Reading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    # Take the first 10 songs
    first_10 = df.head(10)
    
    success_count = 0
    missing_preview_count = 0

    print("Fetching audio previews for the first 10 songs...\n")
    for index, row in first_10.iterrows():
        track_id = row['track_id']
        track_name = row['track_name']
        artist_name = row['track_artist']
        
        try:
            track_info = sp.track(track_id)
            preview_url = track_info.get("preview_url")
            
            if preview_url:
                print(f"[{index+1}/10] Downloading '{track_name}' by {artist_name}...")
                safe_name = "".join([c for c in f"{track_name} - {artist_name}" if c.isalpha() or c.isdigit() or c==' ']).rstrip()
                file_path = os.path.join(output_dir, f"{safe_name}.mp3")
                
                if download_audio(preview_url, file_path):
                    print(f"    -> Saved to {file_path}")
                    success_count += 1
                else:
                    print(f"    -> Failed to download audio from {preview_url}")
            else:
                print(f"[{index+1}/10] No preview URL available for '{track_name}' by {artist_name}.")
                missing_preview_count += 1
                
        except Exception as e:
            print(f"[{index+1}/10] Failed to fetch track '{track_name}' by {artist_name}. Error: {e}")
        
        # Slight delay to avoid hitting rate limits
        time.sleep(1)

    print(f"\nDone! Downloaded {success_count} files. {missing_preview_count} tracks had no preview available.")

if __name__ == "__main__":
    main()
