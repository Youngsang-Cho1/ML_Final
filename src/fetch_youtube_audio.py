import pandas as pd
import yt_dlp
import os
import time

def main():
    csv_path = "data/dataset/spotify_songs.csv"
    output_dir = "data/audio_files"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading dataset from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return

    first_1000 = df.head(1000)
    success_count = 0
    fail_count = 0

    print("Fetching Youtube audio for the first 1000 songs...\n")
    
    for index, row in first_1000.iterrows():
        track_name = row['track_name']
        artist_name = row['track_artist']
        safe_name = "".join([c for c in f"{track_name} - {artist_name}" if c.isalpha() or c.isdigit() or c==' ']).rstrip()
        file_path = os.path.join(output_dir, f"{safe_name}.m4a")
        
        if os.path.exists(file_path):
            print(f"[{index+1}/1000] Skipping '{track_name}' by {artist_name} -> '{safe_name}.m4a' already exists.")
            success_count += 1
            continue
            
        print(f"[{index+1}/1000] Downloading '{track_name}' by {artist_name} from YouTube...")
        query = f"ytsearch1:{track_name} {artist_name} audio"
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]', # Force M4A format so no FFMPEG conversion is needed
            'outtmpl': file_path,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.extract_info(query, download=True)
                print(f"    -> Saved directly as {file_path}")
                success_count += 1
        except Exception as e:
            print(f"    -> Failed to download audio. Error: {str(e)[:200]}")
            fail_count += 1
            
        time.sleep(1)
        
    print(f"\nDone! Successfully downloaded {success_count} files. {fail_count} failures.")

if __name__ == "__main__":
    main()
