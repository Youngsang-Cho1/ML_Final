import os
import glob

def main():
    audio_dir = "data/audio_files"
    spectrogram_dir = "data/spectrograms"
    
    print("Starting cleanup of temporary storage files...")
    
    # Clean up audio files
    if os.path.exists(audio_dir):
        audio_files = glob.glob(os.path.join(audio_dir, "*.m4a"))
        audio_files.extend(glob.glob(os.path.join(audio_dir, "*.mp3")))
        for f in audio_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error removing {f}: {e}")
        print(f"Removed {len(audio_files)} audio files from {audio_dir}.")
    else:
        print(f"Directory {audio_dir} not found.")

    # Clean up spectrogram images
    if os.path.exists(spectrogram_dir):
        image_files = glob.glob(os.path.join(spectrogram_dir, "*.png"))
        for f in image_files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Error removing {f}: {e}")
        print(f"Removed {len(image_files)} spectrogram images from {spectrogram_dir}.")
    else:
        print(f"Directory {spectrogram_dir} not found.")

    print("Cleanup complete! Only the vector embeddings remain.")

if __name__ == "__main__":
    main()
