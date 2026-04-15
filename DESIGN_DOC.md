# Design Document: Multimodal Music Discovery Engine
**CSCI-UA 473: Fundamentals of Machine Learning | Spring 2026**

## 1. Project Proposal

### Problem Description
Current music recommendation systems (like Spotify's "Discover Weekly") often rely heavily on collaborative filtering or artist-based metadata, which can lead to a "filter bubble" where users only hear familiar genres. Our project addresses the problem of **Cold-Start Sentiment-based Music Discovery**. How can we help users find new music based purely on the "sonic signature" (audio vibe) and cross-modal descriptions (textual mood) without relying on past listening history? We use the "30k Spotify Songs" dataset to build a system that bridges the gap between text-based mood requests and raw audio characteristics.

### Methodology
To solve this, we implement a multimodal pipeline consisting of:
1.  **Stratified Random Sampling:** To avoid genre bias (e.g., all samples being 'Pop'), we implement a sampling layer that ensures an equal distribution of songs across all 6 major Spotify genres (Pop, Rap, Rock, Latin, R&B, EDM).
2.  **Representation Learning:** We convert raw audio signals into music-native features using **Librosa MFCCs**, Chroma, and Spectral features (56D). This replaces generic image-based DINOv2 embeddings for better musical alignment.
3.  **Dimensionality Reduction:** We apply **Separate PCA** to fuse 13D metadata and 56D audio features into a condensed 25D space, preventing signal dominance.
4.  **Clustering:** We use a **Manual NumPy Implementation of K-Means** (Lloyd's Algorithm) to group songs into "sonic neighborhoods." This satisfies the requirement for a non-wrapper algorithm implementation.
5.  **Model Validation:** We evaluate cluster quality using the **Elbow Method (WCSS)** and **Silhouette Scores**, providing an interactive optimization UI for the user.
6.  **Hybrid Retrieval Logic:** We implement a **Two-Stage Hybrid Search**:
    -   *Stage 1 (Keyword Filtering):* LLM extracts categorical constraints (e.g., "K-Pop", "Jazz") for hard-filtering the dataset.
    -   *Stage 2 (Vector Ranking):* Euclidean distance matching is performed only on the filtered subset for high semantic precision.
7.  **On-Demand Playback:** A real-time YouTube playback system using `yt-dlp` allows users to preview recommendations directly in the dashboard.

---

## 2. Repository Structure

```text
ML_Final/
├── app/
│   └── streamlit_app.py      # Main full-stack dashboard (UI/UX)
├── data/
│   ├── dataset/              # Final processed parquets and similarity matrices
│   └── embeddings/           # Raw DINOv2 audio embedding vectors
├── src/
│   ├── balance_dataset.py    # Stratified Random Sampling for genre balance
│   ├── build_master_dataset.py # Script for SQL-style joins and data cleaning
│   ├── pca.py                # Implementation of PCA for multimodal fusion
│   ├── clustering.py         # Manual K-Means implementation & Evaluation metrics
│   ├── cosine_similarity.py  # From-scratch implementation of similarity matrix
│   ├── query_parsers.py      # LLM and Audio (Librosa) retrieval encoders (Hybrid Logic)
│   └── audio_utils.py        # YouTube integration and playback caching
├── PIPELINE_EXPLAINED.md     # Technical deep-dive for graders
├── DESIGN_DOC.md             # This document (NYU Rubric Compliance)
└── requirements.txt          # Python dependencies
```

---

## 3. Division of Labor (Team Size: 5)

| Team Member | Primary Responsibility | Key Deliverable |
| :--- | :--- | :--- |
| **Aiden (영원)** | Project Lead & Backend | Fusion pipeline (PCA), Master Dataset logic, Git Management. |
| **[NAME 2]** | ML Algorithm Implementation | K-Means clustering logic and manual Cosine Similarity implementation. |
| **[NAME 3]** | Audio Data Engineering | MFCC feature extraction and Parquet management. |
| **Kai** | Full-Stack UI/UX | Playback System integration, YouTube downloads, and recommendation UI. |

---

## 4. Stub Code & Submission
- **GitHub Repository:** Publicly available with all modules linked.
- **Requirements:** Inclusive of `torch`, `transformers`, `langchain`, `librosa`, and `plotly`.
- **Environment:** Clean `.venv` setup provided.
