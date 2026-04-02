import os
import glob
import librosa
import numpy as np
import matplotlib
# Use Agg backend for headless environments
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_spectrogram(audio_path, output_path):
    """
    Load an audio file and save its Mel spectrogram as an image without any axes.
    """
    # Load the audio (downmixes to mono automatically, resamples to 22050 Hz)
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Compute Mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    
    # Convert power to decibels (log scale)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    # Set up matplotlib figure
    # We use a specific figsize but no frame/axes to get a clean image 
    plt.figure(figsize=(10, 4), frameon=False)
    plt.axis('off')
    
    # Draw the spectrogram using imshow directly or specshow
    # We use specshow which properly maps the 2D array
    librosa.display.specshow(S_dB, sr=sr, fmax=8000, cmap='viridis')
    
    # Save directly to file without white borders
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close()

def main():
    input_dir = "data/audio_files"
    output_dir = "data/spectrograms"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    audio_files = glob.glob(os.path.join(input_dir, "*.mp3")) + glob.glob(os.path.join(input_dir, "*.m4a"))
    
    if not audio_files:
        print(f"No audio files found in '{input_dir}'. Please fetch spotify audio first.")
        return
        
    print(f"Found {len(audio_files)} audio files. Generating spectrogram images...")
    
    for i, audio_path in enumerate(audio_files, 1):
        if audio_path.endswith('.mp3'):
            new_path = audio_path[:-4] + '.m4a'
            os.rename(audio_path, new_path)
            audio_path = new_path
            
        filename = os.path.basename(audio_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}.png")
        
        print(f"[{i}/{len(audio_files)}] Processing '{filename}' -> '{name}.png'")
        try:
            generate_spectrogram(audio_path, output_path)
        except Exception as e:
            print(f"    -> Error processing {filename}: {e}")
            
    print("Spectrogram generation complete!")

if __name__ == "__main__":
    main()
