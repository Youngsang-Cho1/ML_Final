# ML Final Project

## Project Objective
The problem users face is that they often know the type of songs they want, but struggle to find more songs that match their preferences. Songs from different general labels such as genre or artist can still feel similar since their audio features might be similar, while songs from the same category can also sound different. Our goal is to build a music discovery web app that helps users find songs similar to a selected song or ones that match their preferences more directly based on numerical audio features rather than just general labels.

## Dataset
We are using the 30,000 Spotify Songs dataset from Kaggle. It contains 23 variables across three main categories:
- Track metadata: Song name, artist.
- Playlist information: Playlist genre, subgenre.
- Numerical audio features: Danceability, energy, valence, tempo, etc.

## Methodology
The web app uses methods and algorithms covered in our Machine Learning course:
- K-Means Clustering: We map each song into a vector space using its 12 numerical audio features. K-Means clustering then groups songs with similar audio characteristics into different categories. 
- Cosine Similarity: Within a specific cluster, we apply cosine similarity to find songs that are most similar to a user’s selected song or preferred characteristics. The similarity scores rank the songs to provide the closest matches as recommendations based on actual sound rather than abstract labels.
- PCA (Principal Component Analysis): Since 12 numerical features in a high-dimensional space can be difficult to interpret, we apply PCA to reduce dimensionality. This transforms the features into a smaller number of principal components while preserving maximum variance, allowing us to evaluate and interpret our clusters more effectively.

## Deliverables
- Codebase: A GitHub repository containing deployable code.
- Demo: A deployed web application built with Streamlit.
- Presentation: A live demonstration to be presented during the final weeks of the course.

## Repository Structure
- data/: Contains raw and processed datasets.
- src/: Source code for the modules (preprocessing, clustering, similarity, PCA, and recommendation system).
- app/: Contains the Streamlit web application.
- tests/: Unit tests for the modules.

## Setup Instructions
1. Create a virtual environment using `uv`: `uv venv`
2. Activate the python virtual environment: `source .venv/bin/activate`
3. Install the necessary dependencies: `uv pip install -r requirements.txt spotipy pandas requests python-dotenv`

## Updating Dataset Audio
If you'd like to download audio locally into `.m4a` files using YouTube and embed them using Hugging Face DINOv2:
1. Run the fetching script (uses `yt-dlp` to query matches directly from YouTube without needing any API keys!):
   ```sh
   python src/fetch_youtube_audio.py
   ```
   This will read the first 1,000 tracks from `data/dataset/spotify_songs.csv` and download the `.m4a` audio directly into the `data/audio_files/` directory.

3. **Convert to Spectrograms**:
   Once you have downloaded the audio preview files, you can generate image spectrograms for them by running:
   ```sh
   python src/generate_spectrograms.py
   ```
   This will read the `.m4a`/`.mp3` files in `data/audio_files/`, create Mel Spectrogram images without borders or axes, and save them as `.png` files in the `data/spectrograms/` directory.

4. **Embed Spectrograms**:
   Once the images are generated, generate their embeddings using DINOv2:
   ```sh
   python src/spectrogram_embedding.py
   ```
   This reads the `.png` files, evaluates them with the `facebook/dinov2-small` model running locally, and creates a vectorized dataset stored as `data/embeddings/embedded_parquet.parquet`.
