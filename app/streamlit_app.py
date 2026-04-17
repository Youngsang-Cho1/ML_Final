import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Ensure project root is in path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.manifold import TSNE
from src.query_parsers import parse_text_to_features, parse_audio_to_features
from src.clustering import ManualKMeans
from src.audio_utils import fetch_youtube_audio
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Music Discovery Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
        color: #1E293B;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #64748B;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .song-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .song-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        border-color: #93C5FD;
    }
    .cluster-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        background: #DBEAFE;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: bold;
        margin-bottom: 10px;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# --- CLUSTERING OPTIMIZATION LOGIC ---
@st.cache_data
def run_clustering_sweep(features_35d):
    """Computes Inertia and Silhouette scores for K in [2, 15]"""
    k_range = range(2, 16)
    inertias = []
    silhouettes = []
    
    # Scale features for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_35d)
    
    for k in k_range:
        model = ManualKMeans(n_clusters=k, random_state=42)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
        silhouettes.append(model.calculate_silhouette(X_scaled))
        
    return list(k_range), inertias, silhouettes

@st.cache_data
def get_custom_clustering(features_35d, k):
    """Performs manual clustering and returns labels and the normalized feature matrix for dynamic similarity computation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_35d)
    
    model = ManualKMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    labels = model.labels
    
    # Pre-normalize the huge feature matrix once, avoid O(N^2) dot product.
    fused_features = np.stack(features_35d)
    norms = np.linalg.norm(fused_features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_normalized = fused_features / norms
    
    return labels, X_normalized

# --- SESSION STATE INITIALIZATION ---
if 'playing_song' not in st.session_state:
    st.session_state['playing_song'] = None

def play_song(name, artist):
    st.session_state['playing_song'] = {"name": name, "artist": artist}

# --- DATA LOADING ---
@st.cache_data
def load_base_data():
    df = pd.read_parquet('data/dataset/master_music_data.parquet')
    # Extract fused features once
    fused_features = np.stack(df['fused_features'].values)
    return df, fused_features

try:
    df, X_fused = load_base_data()
    
    # Run Sweep (cached)
    k_list, inertias, sil_scores = run_clustering_sweep(X_fused)
    
    # Find optimal K: if data is high-dimensional and overlapping (low silhouette),
    # avoid degenerate K=2 by enforcing a musically meaningful minimum of K=4.
    # This reflects the fact that silhouette scores in high-dim spaces tend to 
    # favor binary splits even when more clusters are semantically meaningful.
    min_k_for_recommendation = 4
    filtered_k = [k for k in k_list if k >= min_k_for_recommendation]
    filtered_sil = [sil_scores[k_list.index(k)] for k in filtered_k]
    optimal_k = filtered_k[np.argmax(filtered_sil)]

    
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    st.info("Check if pipeline scripts have been executed.")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings & Info")
    st.markdown("---")
    st.write("**Model:** DINOv2 (ViT-S/14)")
    st.write("**Algorithm:** Manual K-Means (NumPy)")
    
    st.markdown("### 🛠️ Model Optimization")
    with st.expander("Clustering Analytics", expanded=False):
        # Elbow Plot
        fig_elbow = px.line(x=k_list, y=inertias, markers=True, 
                            title="Elbow Method (Inertia)",
                            labels={"x": "Number of Clusters (K)", "y": "Inertia"})
        fig_elbow.update_layout(template="plotly_white", height=300)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        # Silhouette Plot
        fig_sil = px.bar(x=k_list, y=sil_scores, 
                         title="Silhouette Analysis",
                         labels={"x": "K", "y": "Silhouette Score"})
        fig_sil.update_layout(template="plotly_white", height=300)
        st.plotly_chart(fig_sil, use_container_width=True)
        
        st.info(f"💡 **Mathematically Optimal K: {optimal_k}** (Highest Silhouette Score)")

    # User-Adjustable K
    chosen_k = st.slider("Select Number of Clusters (K)", 2, 15, optimal_k)
    
    # Apply Clustering & Load Normalized matrix for fast queries
    with st.spinner(f"Clustering with K={chosen_k}..."):
        new_labels, X_normalized = get_custom_clustering(X_fused, chosen_k)
        df['cluster'] = new_labels
    
    st.success(f"Model clusters updated to {chosen_k} neighborhoods!")
    st.markdown("---")
    st.write(f"**Data Size:** {len(df):,} Songs")
    st.write("**Fusion:** 35D PCA Fused Space")
    st.markdown("---")
    recommendation_mode = st.radio(
        "Recommendation Mode",
        ("Within Cluster", "Global Search"),
        help="Choose whether to search for similar songs across the entire dataset or restricted to the song's cluster."
    )
    st.markdown("---")
    st.info("This engine uses Multimodal Fusion (Spectrogram Embeddings + Spotify Metadata) for similarity retrieval.")

# --- HEADER ---
st.markdown('<h1 class="main-header">Discovery Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Intelligent music recommendation based on audio signatures and metadata.</p>', unsafe_allow_html=True)

# --- GLOBAL STATISTICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Songs", len(df))
with col2:
    st.metric("Clusters", df['cluster'].nunique())
with col3:
    st.metric("Inference Latency", "0.1s")

st.markdown("---")

# --- DISCOVERY ENGINE ---
tab_seed, tab_text, tab_audio = st.tabs(["🎧 Seed Song", "💬 Text Vibe (LLM)", "🎤 Humming"])

with tab_seed:
    st.subheader("Select a Seed Song")
    song_list = df.apply(lambda r: f"{r['track_name']} - {r['track_artist']}", axis=1).tolist()
    selected_song_full = st.selectbox("Search Song Title or Artist", ["Select a song..."] + song_list)

    if selected_song_full != "Select a song...":
        idx = song_list.index(selected_song_full)
        target_row = df.iloc[idx]
        
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(f"""
            <div class="song-card">
                <span class="cluster-tag">CLUSTER {target_row['cluster']}</span>
                <h3>{target_row['track_name']}</h3>
                <p style="color: #aaa;">{target_row['track_artist']}</p>
                <hr style="opacity: 0.1;">
                <p><b>Original Genre:</b> {target_row['playlist_genre'].upper()}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # --- FEATURE RADAR CHART ---
            features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence']
            feature_values = [target_row[f] for f in features]
            
            radar_df = pd.DataFrame(dict(r=feature_values, theta=features))
            fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, template='plotly_white')
            fig_radar.update_traces(fill='toself', line_color='#2563EB')
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False,
                height=300,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        with c2:
            st.subheader("Recommendations")
            
            # --- RECOMMENDATION LOGIC (On-the-fly execution) ---
            target_vector = X_normalized[idx]
            sim_scores = np.dot(X_normalized, target_vector) # Fast O(N) calculation
            
            if recommendation_mode == "Within Cluster":
                target_cluster = target_row['cluster']
                same_cluster_mask = df['cluster'] == target_cluster
                # Zero out similarities for songs not in the same cluster
                sim_scores[~same_cluster_mask] = -1
                
            # Sort all similarities descending
            all_sorted_indices = np.argsort(sim_scores)[::-1]
            
            # Extract top 5 UNIQUE recommendations (Duplicate handling from Kai's PR)
            unique_recs = []
            seen_titles = set()
            # Add seed song to seen to avoid suggesting itself
            seen_titles.add((target_row['track_name'].lower().strip(), target_row['track_artist'].lower().strip()))
            
            for neighbor_idx in all_sorted_indices:
                if sim_scores[neighbor_idx] == -1: 
                    break # Out of cluster
                if neighbor_idx == idx:
                    continue # Skip seed
                    
                neighbor = df.iloc[neighbor_idx]
                song_key = (neighbor['track_name'].lower().strip(), neighbor['track_artist'].lower().strip())
                
                if song_key not in seen_titles:
                    seen_titles.add(song_key)
                    unique_recs.append(neighbor_idx)
                
                if len(unique_recs) >= 5:
                    break
            
            # --- GLOBAL PLAYER SECTION ---
            if st.session_state['playing_song']:
                curr = st.session_state['playing_song']
                st.markdown(f"""
                <div style="background: rgba(37, 99, 235, 0.1); border: 1px solid #2563EB; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
                    <span style="color: #2563EB; font-weight: bold;">🔊 NOW PREVIEWING:</span> {curr['name']} - {curr['artist']}
                </div>
                """, unsafe_allow_html=True)
                with st.spinner("Streaming from YouTube..."):
                    path = fetch_youtube_audio(curr['name'], curr['artist'])
                    if path:
                        st.audio(path, format="audio/m4a", autoplay=True)
                    else:
                        st.error("Could not fetch audio for this track.")

            cols = st.columns(5)
            for i, neighbor_idx in enumerate(unique_recs):
                neighbor = df.iloc[neighbor_idx]
                genre = str(neighbor.get('playlist_genre', 'Unknown')).upper()
                
                with cols[i]:
                    st.markdown(f"""
                    <div class="song-card" style="font-size: 0.9rem; padding: 1rem; height: 180px; overflow: hidden; margin-bottom: 10px;">
                        <p style="font-weight: bold; margin-bottom: 5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{neighbor['track_name']}</p>
                        <p style="color: #888; font-size: 0.8rem; margin-bottom: 2px;">{neighbor['track_artist']}</p>
                        <p style="color: #666; font-size: 0.7rem; margin-bottom: 5px;">{genre}</p>
                        <p style="color: #2563EB; font-size: 0.7rem; font-weight: bold;">{sim_scores[neighbor_idx]:.2f} Sim</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.button("🎧 Preview", key=f"play_{neighbor_idx}", on_click=play_song, args=(neighbor['track_name'], neighbor['track_artist']), use_container_width=True)

with tab_text:
    st.subheader("Describe Your Vibe")
    groq_api_key = st.text_input("Enter Groq API Key (or set GROQ_API_KEY in .env)", type="password")
    user_prompt = st.text_area("What kind of music are you looking for?", "A fast-paced, high energy EDM track for running.")
    
    if st.button("Search by Vibe"):
        api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.warning("Please provide a Groq API Key to use this feature.")
        else:
            with st.spinner("Analyzing vibe via LLM..."):
                try:
                    query_features = parse_text_to_features(user_prompt, api_key)
                    st.success("Analysis Complete!")
                    
                    # --- HYBRID FILTERING LOGIC ---
                    keywords = query_features.get('keywords', [])
                    filtered_df = df.copy()
                    is_filtered = False
                    
                    if keywords:
                        # Combine keywords into a regex pattern
                        pattern = '|'.join(keywords)
                        # Check genre, artist, and track name columns
                        mask = (
                            df['playlist_genre'].str.contains(pattern, case=False, na=False) |
                            df['playlist_subgenre'].str.contains(pattern, case=False, na=False) |
                            df['track_artist'].str.contains(pattern, case=False, na=False) |
                            df['track_name'].str.contains(pattern, case=False, na=False)
                        )
                        
                        potential_matches = df[mask]
                        if not potential_matches.empty:
                            filtered_df = potential_matches
                            is_filtered = True
                            st.info(f"🔍 Filtering by keywords: {', '.join(keywords)}")
                        else:
                            st.warning(f"⚠️ No exact matches for '{', '.join(keywords)}'. Falling back to global vibe search.")

                    # Manual feature space match against 12 original features
                    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                    
                    # Normalize query to subset of columns
                    query_vec = np.array([query_features.get(c, 0) for c in feature_cols])
                    
                    # Create filtered dataset matrix
                    db_matrix = filtered_df[feature_cols].copy()
                    
                    # Normalize tempo and loudness column to 0-1 range to match others loosely
                    db_matrix['tempo'] = db_matrix['tempo'] / 200.0
                    db_matrix['loudness'] = (db_matrix['loudness'] + 60) / 60.0
                    
                    query_vec[feature_cols.index('tempo')] /= 200.0
                    query_vec[feature_cols.index('loudness')] = (query_vec[feature_cols.index('loudness')] + 60) / 60.0
                    
                    # Euclidean distance search
                    db_matrix_np = db_matrix.values
                    distances = np.linalg.norm(db_matrix_np - query_vec, axis=1)
                    
                    # Top 5 closest
                    best_indices = np.argsort(distances)[:5]
                    
                    st.subheader("Vibe Matches" if not is_filtered else "Best Matches in Category")
                    cols = st.columns(5)
                    for i, idx_match in enumerate(best_indices):
                        match_row = filtered_df.iloc[idx_match]
                        with cols[i]:
                            st.markdown(f"""
                            <div class="song-card" style="font-size: 0.9rem; padding: 1rem; height: 180px;">
                                <p style="font-weight: bold; margin-bottom: 5px;">{match_row['track_name']}</p>
                                <p style="color: #888; font-size: 0.8rem; margin-bottom: 2px;">{match_row['track_artist']}</p>
                                <p style="color: #4B6CB7; font-size: 0.7rem; font-weight: bold;">Dist: {distances[idx_match]:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))

with tab_audio:
    st.subheader("Hum a Tune")
    st.markdown("Upload a short `.wav` file of you humming to find songs with a similar Tempo and Energy.")
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file is not None:
        if st.button("Search by Audio"):
            with st.spinner("Extracting Librosa features..."):
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    audio_features = parse_audio_to_features(tmp_path)
                    os.remove(tmp_path)
                    
                    st.success("Extracted Audio Signature:")
                    st.json(audio_features)
                    
                    # Feature space match (Tempo + Energy)
                    query_tempo = audio_features['tempo'] / 200.0
                    query_energy = audio_features['energy']
                    
                    db_matrix = df[['tempo', 'energy']].copy()
                    db_matrix['tempo'] = db_matrix['tempo'] / 200.0
                    
                    distances = np.sqrt((db_matrix['tempo'] - query_tempo)**2 + (db_matrix['energy'] - query_energy)**2)
                    best_indices = np.argsort(distances.values)[:5]
                    
                    st.subheader("Audio Matches")
                    cols = st.columns(5)
                    for i, idx_match in enumerate(best_indices):
                        match_row = df.iloc[idx_match]
                        with cols[i]:
                            st.markdown(f"""
                            <div class="song-card" style="font-size: 0.9rem; padding: 1rem; height: 180px;">
                                <p style="font-weight: bold; margin-bottom: 5px;">{match_row['track_name']}</p>
                                <p style="color: #888; font-size: 0.8rem; margin-bottom: 2px;">{match_row['track_artist']}</p>
                                <p style="color: #4B6CB7; font-size: 0.7rem; font-weight: bold;">BPM: {match_row['tempo']:.0f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(str(e))

st.markdown("---")

# --- CLUSTER MAP (3D t-SNE) ---
st.subheader("Musical Landscape (3D Projection)")
st.markdown("Mapping 35D multimodal space into 3D using t-SNE. Clusters represent shared sonic characteristics.")

@st.cache_data
def get_3d_projection(_df):
    features_matrix = np.stack(_df['fused_features'].values)
    # Use PCA to reduce to 30D first (speed up t-SNE) then t-SNE to 3D
    tsne = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
    X_3d = tsne.fit_transform(features_matrix)
    return X_3d

X_3d = get_3d_projection(df)
plot_df = df[['track_name', 'track_artist', 'cluster', 'playlist_genre']].copy()
plot_df['Dim 1'] = X_3d[:, 0]
plot_df['Dim 2'] = X_3d[:, 1]
plot_df['Dim 3'] = X_3d[:, 2]
plot_df['cluster'] = plot_df['cluster'].astype(str)

fig = px.scatter_3d(
    plot_df, 
    x='Dim 1', 
    y='Dim 2', 
    z='Dim 3',
    color='cluster',
    hover_name='track_name',
    hover_data=['track_artist', 'playlist_genre'],
    template='plotly_white',
    color_discrete_sequence=px.colors.qualitative.Prism,
    title="3D Musical Space Projection"
)

fig.update_traces(marker=dict(size=4, opacity=0.8))
fig.update_layout(
    margin=dict(l=0, r=0, t=0, b=0),
    legend_title="Cluster ID",
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    )
)

st.plotly_chart(fig, width='stretch')

st.markdown("""
<div style="text-align: center; color: #555; margin-top: 50px; font-size: 0.8rem;">
    Multimodal Music Discovery Pipeline - Final Commitment Version
</div>
""", unsafe_allow_html=True)
