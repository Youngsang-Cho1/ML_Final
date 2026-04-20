# 🎵 Multimodal Music Discovery Engine

**Course:** CSCI-UA 473: Fundamentals of Machine Learning (New York University | Spring 2026)  
**Instructor:** Prof. Kyunghyun Cho  

---

## 📖 Project Objective
Current music recommendation systems often face the **"Filter Bubble"** problem, relying heavily on collaborative filtering or artist metadata. High-level tags like "Pop" or "Rock" are too subjective to capture the actual sonics of a track.

Our objective is to solve the **Cold-Start Sentiment-based Music Discovery** problem. We map songs into a mathematically sound, 45-dimensional **Multimodal Space** (11D Metadata + 34D Audio) that captures 95% of the information coverage. This allows users to find songs that share the exact same *"sonic neighborhood"* regardless of their assigned genre or popularity.

---

## 🧠 Core Machine Learning Algorithms (Rubric Compliance)
This project explicitly implements core machine learning components from scratch, without relying on black-box wrappers (like `sklearn.cluster.KMeans`), to demonstrate algorithmic mastery:

1. **Manual K-Means Clustering**: 
   - A pure NumPy implementation of Lloyd's Algorithm utilizing the dot-product identity for scaling.
   - Evaluated using both Inertia (Elbow Method) and Silhouette clustering scores.
2. **Feature Fusion via PCA**: 
   - Uses Principal Component Analysis to fuse 13D metadata and 58D audio features into a dense **45D vector space** (95% variance coverage), preventing any single feature source from dominating the metric space.
3. **Manual Cosine Similarity**: 
   - Efficient matrix-multiplication-based sequence for calculating nearest neighbors inside clusters.
4. **LLM Hybrid Retrieval (Cross-modal)**:
   - Utilizes Llama-3 (via Groq) to parse natural language queries into the numerical bounds of our multimodal space, allowing users to search by "Vibe" rather than artists.

---

## ✨ Key Features
- **🎧 Interactive Seed Recommendation**: Select any mapped song to analyze its features on a radar chart and immediately listen to nearest neighbors via real-time YouTube playback natively integrated into Streamlit.
- **💬 Vibe Search**: Enter queries like *"An energetic, fast-paced electronic track for running"* and let the LLM map your text into acoustic data points.
- **🔍 Real-time Inference (Analyze Any Song)**: Search an entirely new song on YouTube that isn't in our database. The system will download it, extract 58 dimensions of `librosa` features, project it onto our saved PCA models, and classify it into an exact matching cluster in less than 15초!

---

## 🚀 Setup Instructions

### 1. Requirements
This project uses several data-heavy audio libraries. Ensure you have activated your Python environment (Conda or `uv` venv) and are on Python 3.9+.

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys
To use the "Text Vibe (LLM)" search tab, you must configure a Groq API Key (the free tier works perfectly).
1. Create a `.env` file in the root directory.
2. Add your key:
```text
GROQ_API_KEY="gsk_your_key_here"
```
*(Note: If you don't use a `.env` file, you can explicitly type your key into the Streamlit UI).*

---

## 💻 Running the Dashboard
To boot up the unified frontend application:

```bash
python -m streamlit run app/streamlit_app.py
```
This will open the dashboard at `http://localhost:8501`.

---

## 🛠️ Data Pipeline Execution (Optional)
If you wish to recreate the models or process brand-new songs in batch, run the pipeline scripts in the following exact sequence:

1. **Audio Feature Extraction**: Reads `.m4a`/`.mp3` files in `data/audio_files/` and converts them to 58D Librosa features.
   ```bash
   python src/audio_feature_extractor.py
   ```
2. **Dimensionality Reduction & Fusion**: Merges extracted features with `spotify_songs.csv` and compresses it to a 35D space. Saves `models/*.pkl` for real-time inference.
   ```bash
   python src/pca.py
   ```
3. **Clustering & Neighborhood Mapping**: Maps the songs to K-Means clusters and generates `clustered_songs.parquet`.
   ```bash
   python src/clustering.py
   ```

---

## 👥 Division of Labor

| Team Member | Primary Responsibility | Key Deliverables |
| :--- | :--- | :--- |
| **Aiden** | Project Lead & Backend | Fusion pipeline (PCA), Master Dataset logic, Serialization. |
| **[NAME 2]** | ML Algorithm Implementation | K-Means clustering logic and manual Cosine Similarity implementation. |
| **[NAME 3]** | Audio Data Engineering | `Librosa` Feature extraction and Parquet memory management. |
| **Kai** | Full-Stack UI/UX | Playback System integration, YouTube `yt-dlp` bridging, and Streamlit recommendation UI. |
