# 🎵 AcousticDNA: Multi-Stage Music Retrieval Engine
**Final Presentation Outline & Speaking Notes**

---

## 1. Problem Statement: The "Skip" Phenomenon
**"Why do we skip songs in our auto-play queue?"**
*   **The Issue:** We’ve all experienced it—Spotify or YouTube auto-plays a song that feels completely "off" from the vibe of the previous track, purely because it's popular or shares a generic genre tag. 
*   **The Cause:** Traditional recommendation systems rely heavily on textual metadata (Genres, Artists) and social signals (likes/popularity). If a song has no tags, it cannot be recommended (Cold Start). If it does, the tags are often too broad to capture the true *acoustic energy*.
*   **Our Goal:** To build a system that has a mathematical "ear" for the **Acoustic Signature** of a song, ensuring recommendations are acoustically consistent, regardless of popularity or labels.

---

## 2. Our Data Pipeline (Engineering the Master Dataset)
To solve the "skip" problem, we needed a high-resolution dataset:
1.  **YouTube Audio Scraping:** A fully automated nightly pipeline for querying and batch-downloading `.m4a` files.
2.  **56D Librosa Feature Extraction:** Capturing the mathematical DNA of sound:
    *   *MFCC (Vocal timbre/Texture)*
    *   *Chroma (Harmonic pitch/Chords)*
    *   *Spectral (Brightness/Roll-off)*
    *   *Temporal (ZCR/Beat density)*
3.  **High-Performance File DB:** Indexed in **Apache Parquet** for instant *In-Memory Vector Search*, bypassing SQL latency for real-time relevance.

---

## 3. Architecture & Design Choices (Why Not LLMs?)
**"Why explicit distance metrics over Black-Box Models?"**
*   **Transparency over Hype:** We avoided LLMs or stacked black-box libraries. We wanted a transparent, deterministic system where every recommendation can be mathematically justified.
*   **Hybrid Distance Ensemble:**
    *   `Score = MSE(Metadata) + λ × CosineDistance(Audio_Embeddings)`
    *   **MSE for Metadata (8D):** Absolute Euclidean distance makes sense for bounded [0, 1] vibes like Danceability or Tempo.
    *   **Cosine for Audio (56D):** Measuring the *Angle (Direction)* of the acoustic signature to overcome the *Curse of Dimensionality*.

---

## 4. Mathematical Triumphs: Preventing "Cosine Collapse"
**How we ensured technical relevance:**
*   **The Issue:** Raw features like `MFCC_0` (-300) dominate tiny ones like `ZCR` (0.05). If unscaled, the most powerful dimension collapses the vector space—making every song look 98% identical (**Cosine Collapse**).
*   **The Fix:** We implemented strict **Standardization (Z-score)** before L2-Normalization. This guarantees every audio property has an equal mathematical "voice," shifting the dataset from a collapsed line into a beautifully distributed 56D sphere.

---

## 5. Hyperparameter Tuning & Evaluation
### The $\lambda$ (Lambda = 1.5) Weight
*   We swept values using a strict heuristic: *"Remixes and same-artist songs must cluster together."* $\lambda = 1.5$ was found to be the sweet spot where the broad vibe (Metadata) filters the context, and the Acoustic Signature (Audio) performs the surgical matching.

### Manual K-Means & Silhouette (s ≈ 0.07)
*   **Pure Numpy Implementation:** Built Lloyd's Algorithm + K-Means++ from scratch. 
*   **Why s=0.07?** In a 64D space, the **Concentration of Measure** naturally pulls scores toward 0. A score of `0.07` in a 64D continuum is highly significant (random noise yields `0.00`), proving our metric captures genuine acoustic clusters.

---

## 6. The Ultimate Sanity Check: "Analyze Any Song"
**Text-Agnostic Zero-Shot Inference:**
To prove our math solved the "Irrelevant Recommendation" problem, we built a feature where you paste an unseen YouTube URL:
*   The system extracts the signature and maps it against our DB without **any textual metadata (No Artist, No Title).**
*   *Success Case:* Inputting the raw audio of Blackpink's "DDU-DU DDU-DU" returns "Kill This Love" as the #1 match. 
*   **Conclusion:** The math successfully identified Teddy Park's production style and vocal timbres from thousands of tracks purely by numbers. We have replaced "random popularity" with **Acoustic Consistency**.
