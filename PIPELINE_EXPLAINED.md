# Multimodal Music Discovery Pipeline: Project Progress & Architecture

This document provides a detailed technical explanation of the "Separate PCA" multimodal fusion pipeline developed for this project.

## Architecture: Multimodal Late Fusion (Separate PCA)
To build a robust recommendation system, we combine two different types of data: **Spotify Tabular Metadata** and **DINOv2 Audio Embeddings**. 

We use a **Late Fusion** approach by applying Dimensionality Reduction (PCA) to each data source *independently* before merging them. This prevents the high-dimensional audio embeddings (384D) from statistically overwhelming the metadata features (13D).

### 1. Spotify Feature Processing
-   **Input**: 13 numeric features (Track Popularity, Danceability, Energy, Key, Loudness, Mode, Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration).
-   **Reduction**: Features are standardized and reduced to **5 Principal Components**.
-   **Loadings Insight**: 
    -   **PC1**: Correlates with "Intensity" (High Energy & Loudness).
    -   **PC2**: Correlates with "Danceability & Positivity".
    -   **PC3**: Correlates with "Musical Scale & Popularity".

### 2. DINOv2 Audio Embedding
-   **Representation**: Audio is converted into **Mel Spectrograms** (visual frequency maps).
-   **Model**: We use `facebook/dinov2-small` (Vision Transformer) to extract a **384-dimensional latent vector** from each spectrogram segment.
-   **Reduction**: The 384 dimensions are reduced to **30 Principal Components** via PCA to remove noise and focus on the most distinct sonic textures.

### 3. Fused Vector & Clustering
-   **Neural Fingerprint**: The 5D metadata and 30D audio components are concatenated to form a **unified 35-dimensional vector**—a high-fidelity representation of the song's identity.
-   **Clustering**: **K-Means** is applied to this 35D space to identify 10 distinct musical neighborhoods.
-   **Advanced Visualization**: Since 35D space is non-perceptual, the application uses **3D t-SNE (t-distributed Stochastic Neighbor Embedding)** to project the manifold into a 3-dimensional interactive map, preserving local cluster structures for the user.

---

## File Roles & Operations

### Phase 1: Data Acquisition & Preprocessing
-   `src/fetch_youtube_audio.py`: Downloads audio for the selected 1,850 songs via YouTube (M4A format).
-   `src/generate_spectrograms.py`: Converts audio to Mel Spectrogram images.
-   `src/spectrogram_embedding.py`: Uses **DINOv2 (ViT-S/14)** to extract 384D latent vectors.
-   `src/cleanup_temp_files.py`: Purges ~3GB of temporary audio/image files once vectorized.

### Phase 2: ML Pipeline (Multimodal Fusion)
-   `src/pca.py`: Implements the **Separate PCA** logic. Standardizes and reduces Spotify (13D->5D) and Audio (384D->30D) separately before fusion.
-   `src/clustering.py`: Executes K-Means on the 35D fused space.
-   `src/cosine_similarity.py`: Pre-calculates the $N \times N$ similarity matrix for ultra-low latency (<0.1s) inference.

### Phase 3: Interactive Demo
-   `app/streamlit_app.py`: The user dashboard featuring:
    -   **Discovery UI**: Search-based retrieval using the pre-cached similarity matrix.
    -   **3D Landscape**: High-dimensional musical space projected into 3D using t-SNE.

---

## Technical Validation
-   **Feature Density**: By fusing DINOv2 (texture) with Spotify metadata (intent), the model correctly identifies cross-genre similarities that metadata alone would miss.
-   **Visualization Accuracy**: Unlike standard 2D PCA, the **3D t-SNE** projection captures the non-linear relationships in the 35D space, allowing users to rotate and explore cluster boundaries interactively.
-   **Performance**: All heavy lifting (Embedding & PCA) is done offline; the web app maintains sub-second responsiveness.

---

## Reproducibility Guide
To regenerate the full pipeline:
1. `python src/fetch_youtube_audio.py`
2. `python src/generate_spectrograms.py`
3. `python src/spectrogram_embedding.py`
4. `python src/cleanup_temp_files.py` (optional but recommended)
5. `python src/pca.py`
6. `python src/clustering.py`
7. `python src/cosine_similarity.py`
8. `streamlit run app/streamlit_app.py`
