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
        
    # We define a fallback chain: SoundCloud -> YouTube Lyrics
    queries = [
        f"scsearch1:{track_name} {artist_name}",
        f"ytsearch1:{track_name} {artist_name} lyrics"
    ]
    
    ydl_opts = {
        'format': 'bestaudio/best',
        # yt-dlp + ffmpeg postprocessor will automatically append .m4a to this outtmpl
        'outtmpl': file_path.replace('.m4a', ''),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'extractor_args': {'youtube': ['player_client=android']}
    }
    
    try:
        import yt_dlp
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for query in queries:
                try:
                    ydl.extract_info(query, download=True)
                    if os.path.exists(file_path):
                        manage_cache_size(cache_dir)
                        return file_path
                except Exception as sub_e:
                    print(f"Skipping blocked/unavailable source for {query}: {sub_e}")
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
