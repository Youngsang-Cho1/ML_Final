import os
import glob
import time

def get_safe_name(track_name, artist_name):
    """
    Standardizes 'Song - Artist' into a filesystem-safe string.
    Ensures that downloaded files and dataset join-keys are identical.
    """
    return "".join([c for c in f"{track_name} - {artist_name}" if c.isalpha() or c.isdigit() or c==' ']).rstrip()

def fetch_youtube_audio(track_name, artist_name, cache_dir="data/playback_cache"):
    """
    On-demand fetcher that finds a song on YouTube and downloads the .m4a 
    for real-time playback in the Streamlit app.
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # 1. Generate consistent filename
    safe_name = get_safe_name(track_name, artist_name)
    file_path = os.path.join(cache_dir, f"{safe_name}.m4a")
    
    # 2. Check Cache: Avoid redundant downloads
    if os.path.exists(file_path):
        return file_path
        
    query = f"ytsearch3:{track_name} {artist_name} lyrics"
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]',
        'outtmpl': file_path,
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False
    }
    
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Extract top 3 results without downloading first
            info = ydl.extract_info(query, download=False)
            if info and 'entries' in info:
                for entry in info['entries']:
                    if not entry: continue
                    try:
                        # Try to download the specific video URL
                        ydl.download([entry['webpage_url']])
                        if os.path.exists(file_path):
                            manage_cache_size(cache_dir)
                            return file_path
                    except Exception as sub_e:
                        print(f"Skipping blocked video: {sub_e}")
                        continue
    except Exception as e:
        print(f"Error fetching audio for {track_name}: {e}")
        return None
    
    return None

def manage_cache_size(cache_dir, max_files=20):
    """Keep the playback cache lean by deleting oldest files."""
    files = glob.glob(os.path.join(cache_dir, "*.m4a"))
    if len(files) > max_files:
        # Sort by modification time (oldest first)
        files.sort(key=os.path.getmtime)
        for i in range(len(files) - max_files):
            try:
                os.remove(files[i])
            except:
                pass
