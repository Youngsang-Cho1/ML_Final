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
1. Activate the python virtual environment: `source venv/bin/activate`
2. Install the necessary packages: `pip install -r requirements.txt`
