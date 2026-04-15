import os
import glob
import librosa
import numpy as np
import pandas as pd

def extract_audio_features(audio_path: str) -> np.ndarray:
    """
    Extract a rich set of music-specific features from an audio file using librosa.
    
    Features extracted (total ~58 dimensions):
    - MFCC (13 coefficients): mean + std = 26D  → timbre / vocal quality
    - Chroma (12 pitches): mean = 12D           → harmonic / key information
    - Spectral Centroid: mean + std = 2D        → brightness of the sound
    - Spectral Bandwidth: mean + std = 2D       → richness of the sound
    - Spectral Rolloff: mean + std = 2D         → high-frequency energy cutoff
    - Zero Crossing Rate: mean + std = 2D       → percussiveness / noisiness  
    - RMS Energy: mean + std = 2D               → loudness / intensity
    - Tempo: 1D                                 → BPM
    - Spectral Contrast (6 bands): mean = 6D    → foreground vs background energy
    
    Returns: numpy array of shape (58,)
    """
    # Load audio file, resample to 22050 Hz for consistency
    y, sr = librosa.load(audio_path, sr=22050, mono=True, duration=60)  # use first 60s

    features = []

    # 1. MFCC (13 coefficients) — core timbre descriptor, inspired by human hearing (Mel scale)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1).tolist())   # 13D
    features.extend(np.std(mfcc, axis=1).tolist())    # 13D

    # 2. Chroma STFT (12 pitch classes) — captures harmonic/tonal content
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1).tolist())  # 12D

    # 3. Spectral Centroid — "center of mass" of spectrum = perceived brightness
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spectral_centroid)))  # 1D
    features.append(float(np.std(spectral_centroid)))   # 1D

    # 4. Spectral Bandwidth — spread around centroid = richness
    spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spectral_bw)))  # 1D
    features.append(float(np.std(spectral_bw)))   # 1D

    # 5. Spectral Rolloff — frequency below which 85% of energy lies = high-freq content
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(rolloff)))  # 1D
    features.append(float(np.std(rolloff)))   # 1D

    # 6. Zero Crossing Rate — how often signal crosses zero = percussiveness
    zcr = librosa.feature.zero_crossing_rate(y=y)
    features.append(float(np.mean(zcr)))  # 1D
    features.append(float(np.std(zcr)))   # 1D

    # 7. RMS Energy — root mean square loudness
    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))  # 1D
    features.append(float(np.std(rms)))   # 1D

    # 8. Tempo (BPM)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.feature.rhythm.tempo(onset_envelope=onset_env, sr=sr)
    features.append(float(tempo[0]))  # 1D

    # 9. Spectral Contrast (6 bands) — foreground vs background energy separation
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1).tolist())  # 7D (6 bands + 1 valley)

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

    out_path = os.path.join(embed_dir, "embedded_spectrograms.parquet")
    df.to_parquet(out_path)
    print(f"Successfully saved MFCC audio features to {out_path}")


if __name__ == "__main__":
    main()
