# 🎵 Multimodal Music Discovery Engine

**Course:** CSCI-UA 473: Fundamentals of Machine Learning (New York University | Spring 2026)
**Instructor:** Prof. Kyunghyun Cho

---

## 📖 Project Objective

A music recommendation engine that answers the question:

> *"Given a song, which songs in the database are most similar — regardless of genre label?"*

We deliberately avoid black-box libraries (like Scikit-Learn) where possible, relying instead on **Pure Numpy implementations** (e.g., K-Means++, SVD-based PCA) to demonstrate fundamental machine learning knowledge. This project focuses on **one well-defined multimodal distance metric**, mathematically rigorous data standardizations, and explicit evaluation.

---

## 🧠 Core Algorithm

We represent each song with two modalities kept **separate**:

1. **Metadata** — 8D Spotify audio features (normalized to [0, 1])
   `danceability, energy, speechiness, acousticness, instrumentalness, liveness, valence, tempo`

2. **Audio Embedding** — 56D Librosa features extracted from the raw audio
   `MFCC (26D) + Chroma (12D) + Spectral (12D) + Temporal (6D)`

### Distance Metric

For a query song *q* and candidate song *c*:

```
score(q, c) = MSE(metadata_q, metadata_c) + λ × (1 − cosine_similarity(emb_q, emb_c))
```

**Lower score = more similar.**

- **MSE on metadata** — metadata is low-dimensional (8D) and bounded to [0, 1], where absolute-value differences (e.g. `energy = 0.9` vs `0.1`) carry real musical meaning.
- **Cosine on embedding** — embedding is high-dimensional (56D), where MSE suffers from the curse of dimensionality. Cosine similarity evaluates the *angle* of the acoustic signature and safely maps similarities in high-D space.

### Preventing "Cosine Collapse"
Raw audio features from `librosa` contain wildly varying magnitudes (e.g., `MFCC_0` is usually between -100 and -300, while `ZCR` is ~0.05). If passed directly into Cosine Similarity, the massive dimension dominates the vector's angle, collapsing the similarity space so all songs appear 99% identical (Cosine Collapse).
**Solution:** We carefully enforce **Z-score Standardization (Zero Mean, Unit Variance) per feature dimension** right before calculating L2 Norms. This ensures all 56 acoustic properties have an equal voice in determining the vector angle, resulting in a beautifully distributed similarity space ranging continuously from `1.0` (Identical) to `-1.0` (Orthogonal/Opposite).
 
### The "Acoustic Signature" (Zero-Text Inference)
Because the audio pipeline is mathematically pure, the **Analyze Any Song from YouTube** feature requires zero text metadata. It can download a completely unknown YouTube URL, extract the 56D representation, project it through the Z-score reference of our database, and reliably surface songs by the *same artist* purely by recognizing their vocal timbre and production style!

---

## ⚖️ λ Tuning (Hyperparameter Selection)

**λ controls how much weight the audio embedding gets relative to metadata.** A fixed, principled value — chosen by evaluation, not handed to the user — is used.

### Method: Same-Artist vs Random-Pair Sweep

1. Sample 100 *same-artist* pairs (positive signal: artists typically maintain consistent musical style).
2. Sample 100 *random-artist* pairs (negative signal: unrelated songs).
3. For each λ ∈ {0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0}:
   - Compute the average score for both groups.
   - Compute **pass rate** — the fraction of (same_artist_pair, random_pair) combinations where the same-artist score is lower.

### Result

| λ | same_avg | random_avg | pass_rate |
|---|---|---|---|
| 0.01 | 0.0348 | 0.0606 | 0.7216 |
| 0.1 | 0.0359 | 0.0619 | 0.7222 |
| **0.2** | **0.0371** | **0.0634** | **0.7239** |
| 0.5 | 0.0406 | 0.0678 | 0.7239 |
| 1.0 | 0.0464 | 0.0751 | 0.7213 |
| 2.0 | 0.0581 | 0.0898 | 0.7118 |
| 5.0 | 0.0931 | 0.1340 | 0.6821 |
| 10.0 | 0.1514 | 0.2075 | 0.6514 |

- **Chosen λ = 0.2** (highest pass rate: 72.4%)
- Random-chance baseline would be 50% — our metric ranks same-artist pairs below random pairs **72% of the time**, indicating the distance metric captures real musical similarity.
- Run `python src/tune_lambda.py` to reproduce.

---

## 📊 Evaluation

Two evaluation methods are exposed in the Streamlit app's **Evaluation** tab:

1. **Manual K-Means Clustering (Quality Sweep)** — Evaluates the dataset by sweeping `K=5~12`, calculating the **Elbow (Inertia)** and **Silhouette Scores** dynamically using a pure Numpy K-Means implementation built from scratch. Uncovers the internal cluster coherence of the distance metrics.
2. **PCA 2D Visualization** — The 56D audio embedding is projected to 2D via our manual SVD-based PCA. A scatter plot lets us visually verify that acoustically similar songs naturally cluster together.
3. **Distance System Sanity Check** — A playground to verify that expected relations holds true (e.g. remix versions always mathematically beat out completely unrelated genres).

---

## 🗂️ File Structure

```
src/
  fetch_youtube_audio.py      # Download audio from YouTube (batch mode)
  audio_feature_extractor.py  # Extract 56D Librosa features from audio files
  build_master_dataset.py     # Join metadata CSV + audio embeddings parquet
  recommender.py              # Core: MSE + cosine hybrid distance metric
  tune_lambda.py              # λ hyperparameter sweep
  pca.py                      # Manual PCA (SVD-based) for visualization
  audio_utils.py              # Helpers (safe_name, YouTube fetcher)

app/
  streamlit_app.py            # UI: Seed Song, Analyze Any Song, Evaluation tabs

data/dataset/
  spotify_songs_full.csv      # Full original dataset (~32k songs, read-only)
  processed_songs.csv         # Cumulatively processed tracks
  download_manifest.csv       # Shuffled batch ordering (stable across runs)
  master_music_data.parquet   # Joined metadata + embeddings (used by the app)

data/embeddings/
  audio_features.parquet      # 56D Librosa feature vectors per track
```

---

## 🚀 Setup

```bash
# Python 3.9+
pip install -r requirements.txt
```

---

## 🎛️ Running the App

```bash
python -m streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`.

### Tabs

- **🎧 Seed Song** — pick a song, get the top 5 most similar tracks using our hybrid metric.
- **🔍 Analyze Any Song** — enter a YouTube search query; the app downloads, extracts features, and finds similar songs in the DB (audio-only, since metadata is not available for new tracks).
- **📊 Evaluation** — PCA 2D scatter and sanity check.

---

## 🛠️ Rebuilding the Dataset

If you want to process additional songs:

```bash
# 1. Download YouTube audio (one batch = 1000 songs)
python src/fetch_youtube_audio.py --batch <N>

# Or use the automated nightly pipeline for batches 1–29
caffeinate -i ./run_nightly_pipeline.sh

# 2. Extract Librosa features (also updates processed_songs.csv)
python src/audio_feature_extractor.py

# 3. Build master dataset (join metadata + embeddings)
python src/build_master_dataset.py

# 4. (optional) Re-tune λ for the new data
python src/tune_lambda.py
```

---

## 👥 Division of Labor

| Team Member | Primary Responsibility |
| :--- | :--- |
| **Aiden** | Data engineering — YouTube pipeline, Librosa feature extraction |
| **Kai** | Streamlit frontend, YouTube playback integration |
| **Max** | Manual PCA (SVD) implementation for visualization |
| **Yanfu** | Distance metric design, λ tuning methodology |
| **Sue** | Evaluation (PCA scatter, sanity check) |
