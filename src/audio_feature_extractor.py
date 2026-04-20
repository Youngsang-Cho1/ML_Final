import os
import glob
import librosa
import numpy as np
import pandas as pd

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extracts a 58-dimensional feature vector from audio using Librosa.
    
    Dimensions:
    - MFCC (26D): mean + std. Captures timbre (the 'texture' of the sound).
    - Chroma (12D): mean. Captures harmonic/musical key information.
    - Spectral Centroid/BW/Rolloff (6D): Captures 'brightness' and 'richness'.
    - Zero Crossing Rate/RMS (4D): Captures percussiveness and loudness.
    - Tempo (1D): Beats per minute (BPM).
    - Spectral Contrast (9D): foreground vs background energy separation.
    """
    # Load audio (mono, 22.05kHz) - limited to first 60s for consistency
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)

    features = []

    # 1. MFCCs - The most common feature in speech/music recognition (Mel-Frequency Cepstral Coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1).tolist())
    features.extend(np.std(mfcc, axis=1).tolist())

    # 2. Chroma - Maps energy into 12 semitones of the musical octave
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1).tolist())

    # 3. Spectral Features - Shape of the frequency distribution
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spectral_centroid)))
    features.append(float(np.std(spectral_centroid)))

    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spectral_bw)))
    features.append(float(np.std(spectral_bw)))

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(rolloff)))
    features.append(float(np.std(rolloff)))

    # 4. Temporal Features - Percussiveness and Loudness
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    # 5. Rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    features.append(float(tempo[0]))

    # 6. Spectral Contrast - Separation of frequency peaks and valleys
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1).tolist())

    return np.array(features, dtype=np.float32)


def main():
    input_dir = "data/audio_files"
    embed_dir = "data/embeddings"
    os.makedirs(embed_dir, exist_ok=True)

    # Gather all audio files
    audio_files = (
        glob.glob(os.path.join(input_dir, "*.m4a")) +
        glob.glob(os.path.join(input_dir, "*.mp3"))
    )

    if not audio_files:
        print(f"No audio files found in '{input_dir}'. Please run fetch_youtube_audio.py first.")
        return

    print(f"Found {len(audio_files)} audio files. Extracting MFCC + Chroma + Spectral features...")

    all_embeddings = []
    track_ids = []

    for i, audio_path in enumerate(audio_files, 1):
        filename = os.path.basename(audio_path)
        # safe_name is the filename without extension (same convention as pca.py)
        track_id = os.path.splitext(filename)[0]

        try:
            print(f"[{i}/{len(audio_files)}] Extracting: {filename}")
            features = extract_audio_features(audio_path)
            all_embeddings.append(features.tolist())
            track_ids.append(track_id)
        except Exception as e:
            print(f"    -> SKIPPED {filename}: {e}")

    if not all_embeddings:
        print("No features were extracted successfully.")
        return

    feature_dim = len(all_embeddings[0])
    print(f"\nExtracted {feature_dim}D feature vector for {len(all_embeddings)} songs.")

    # Save in same format as old DINOv2 parquet so pca.py is compatible
    df = pd.DataFrame({
        "track_id": track_ids,
        "embedding": all_embeddings,
    })

    out_path = os.path.join(embed_dir, "audio_features.parquet")
    df.to_parquet(out_path)
    print(f"Successfully saved MFCC audio features to {out_path}")


if __name__ == "__main__":
    main()
