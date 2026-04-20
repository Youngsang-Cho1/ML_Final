import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys

# Ensure project root is in path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.manifold import TSNE
from src.query_parsers import parse_text_to_features
from src.clustering import ManualKMeans
from src.audio_utils import fetch_youtube_audio
from src.audio_feature_extractor import extract_audio_features
import pickle
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
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.4);
    }
    div.stButton > button[kind="primary"]:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.6);
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
    """
    Evaluates different K values for K-Means using Inertia (Elbow) and Silhouette Scores.
    Helps the user pick the mathematically optimal number of clusters.
    """
    k_range = range(2, 16)
    inertias = []
    silhouettes = []
    
    # Standardize data before clustering
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
    """
    Performs the final K-Means clustering with the user-selected K.
    Also returns a fresh dot-product based Cosine Similarity matrix.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features_35d)
    
    # 1. Clustering
    model = ManualKMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    labels = model.labels
    
    # 2. Similarity Calculation (Manual implementation)
    fused_features = np.stack(features_35d)
    norms = np.linalg.norm(fused_features, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    X_normalized = fused_features / norms
    sim_matrix = np.dot(X_normalized, X_normalized.T)
    
    return labels, sim_matrix

# --- SESSION STATE INITIALIZATION ---
# Manages the global state of the application (e.g., currently playing preview)
if 'playing_song' not in st.session_state:
    st.session_state['playing_song'] = None

def play_song(name, artist):
    st.session_state['playing_song'] = {"name": name, "artist": artist}

# --- DATA LOADING ---
@st.cache_data
def load_base_data():
    """
    Loads the 'Source of Truth' master dataset with all fused features.
    Cached to ensure near-instant page reloads.
    """
    df = pd.read_parquet('data/dataset/master_music_data.parquet')
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
    st.write("**Model:** Librosa (MFCC/Chroma)")
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
    
    # Apply Clustering & Get Similarity Matrix
    with st.spinner(f"Clustering with K={chosen_k}..."):
        new_labels, sim_matrix = get_custom_clustering(X_fused, chosen_k)
        df['cluster'] = new_labels
    
    st.success(f"Model clusters updated to {chosen_k} neighborhoods!")
    
    st.markdown("""
    <div style="background: white; border: 2px solid #3B82F6; padding: 20px; border-radius: 12px; margin-top: 15px;">
        <h3 style="color: #1E3A8A; margin-top: 0; font-family: 'Outfit', sans-serif;">🎯 Search Scope Override</h3>
        <p style="color: #64748B; font-size: 0.95rem; margin-bottom: 10px;">
            Determines whether the <strong>Seed Song</strong> and <strong>Vibe</strong> searches should be limited to matching clusters, or search the entire global space.
        </p>
    """, unsafe_allow_html=True)
    
    recommendation_mode = st.radio(
        "Select Scope Strategy:",
        ("Within Cluster", "Global Search"),
        horizontal=True,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)

# --- HEADER ---
st.markdown('<h1 class="main-header">Discovery Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Intelligent music recommendation based on audio signatures and metadata.</p>', unsafe_allow_html=True)

# --- GLOBAL STATISTICS ---
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Songs", len(df))
with col2:
    st.metric("Data Dimensions", "45D Space")
with col3:
    st.metric("Clusters", df['cluster'].nunique())

st.markdown("---")

# --- GLOBAL PLAYER SECTION ---
if st.session_state.get('playing_song'):
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

# --- DISCOVERY ENGINE ---
tab_seed, tab_text, tab_discovery = st.tabs(["🎧 Seed Song", "💬 Text Vibe (LLM)", "🔍 Analyze Any Song"])

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
            
            # --- NEW: Seed Song Playback ---
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🎧 PLAY SEED SONG (YOUTUBE)", key="play_seed", type="primary", use_container_width=True):
                play_song(target_row['track_name'], target_row['track_artist'])
                st.rerun()
            
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
            
            # --- RECOMMENDATION LOGIC ---
            sim_scores = sim_matrix[idx].copy()
            
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
                    if st.button("Preview", key=f"seed_prev_{neighbor_idx}"):
                        play_song(neighbor['track_name'], neighbor['track_artist'])
                        st.rerun()

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
                    # --- HYBRID FILTERING LOGIC ---
                    keywords = query_features.get('keywords', [])
                    filtered_df = df.copy()
                    is_filtered = False
                    
                    if keywords:
                        pattern = '|'.join(keywords)
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

                    feature_cols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
                    query_vec = np.array([query_features.get(c, 0) for c in feature_cols])
                    db_matrix = filtered_df[feature_cols].copy()
                    db_matrix['tempo'] = db_matrix['tempo'] / 200.0
                    db_matrix['loudness'] = (db_matrix['loudness'] + 60) / 60.0
                    query_vec[feature_cols.index('tempo')] /= 200.0
                    query_vec[feature_cols.index('loudness')] = (query_vec[feature_cols.index('loudness')] + 60) / 60.0
                    
                    db_matrix_np = db_matrix.values
                    distances = np.linalg.norm(db_matrix_np - query_vec, axis=1)
                    best_indices = np.argsort(distances)[:5]
                    
                    st.session_state['text_results'] = {
                        'prompt': user_prompt,
                        'indices': best_indices,
                        'distances': distances,
                        'filtered_df': filtered_df,
                        'is_filtered': is_filtered,
                        'keywords': keywords
                    }
                except Exception as e:
                    st.error(str(e))
    
    if st.session_state.get('text_results') and st.session_state['text_results']['prompt'] == user_prompt:
        res = st.session_state['text_results']
        st.success("Analysis Complete!")
        if res['is_filtered']:
            st.info(f"🔍 Filtering by keywords: {', '.join(res['keywords'])}")
        elif res['keywords']:
            st.warning(f"⚠️ No exact matches for '{', '.join(res['keywords'])}'. Falling back to global vibe search.")
            
        st.subheader("Vibe Matches" if not res['is_filtered'] else "Best Matches in Category")
        cols = st.columns(5)
        for i, idx_match in enumerate(res['indices']):
            match_row = res['filtered_df'].iloc[idx_match]
            with cols[i]:
                st.markdown(f"""
                <div class="song-card" style="font-size: 0.9rem; padding: 1rem; height: 180px;">
                    <p style="font-weight: bold; margin-bottom: 5px;">{match_row['track_name']}</p>
                    <p style="color: #888; font-size: 0.8rem; margin-bottom: 2px;">{match_row['track_artist']}</p>
                    <p style="color: #4B6CB7; font-size: 0.7rem; font-weight: bold;">Dist: {res['distances'][idx_match]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Preview", key=f"txt_prev_{i}"):
                    play_song(match_row['track_name'], match_row['track_artist'])
                    st.rerun()

with tab_discovery:
    st.subheader("Discover & Analyze Any Song")
    st.markdown("Search for any song on YouTube. We will download it, extract its audio features, and map it into our **45D musical space** (95% variance coverage).")
    
    col_input, col_sliders = st.columns([1, 1])
    
    with col_input:
        search_query = st.text_input("YouTube Search", placeholder="e.g. NewJean - Super Shy")
        st.info("💡 Since we don't have Spotify metadata for new songs, please help us tune the 'vibe' below.")

    with col_sliders:
        st.write("**Manual Metadata Tuning**")
        s_valence = st.slider("Valence (Happiness)", 0.0, 1.0, 0.5)
        s_energy = st.slider("Energy", 0.0, 1.0, 0.5)
        s_dance = st.slider("Danceability", 0.0, 1.0, 0.5)
        s_tempo = st.slider("Tempo (BPM estimate)", 60, 180, 120)
        s_pop = st.slider("Popularity", 0, 100, 50)

    if search_query:
        if st.button("🚀 Analyze & Match"):
            with st.spinner("Step 1: Downloading from YouTube..."):
                try:
                    tmp_path = fetch_youtube_audio(search_query, "")
                    st.audio(tmp_path)
                    
                    with st.spinner("Step 2: Extracting 58D Librosa Features..."):
                        raw_audio_features = extract_audio_features(tmp_path)
                    
                    with st.spinner("Step 3: Dimensionality Reduction (PCA)..."):
                        with open('models/audio_pca.pkl', 'rb') as f:
                            a_model = pickle.load(f)
                        with open('models/metadata_pca.pkl', 'rb') as f:
                            m_model = pickle.load(f)
                        with open('models/cluster_model.pkl', 'rb') as f:
                            c_model = pickle.load(f)

                        a_scaled = a_model['scaler'].transform(raw_audio_features.reshape(1, -1))
                        a_pca = a_model['pca'].transform(a_scaled)

                        meta_raw = np.array([[s_pop, s_dance, s_energy, 0, -5, 1, 0.05, 0.1, 0, 0.1, s_valence, s_tempo, 200000]])
                        m_scaled = m_model['scaler'].transform(meta_raw)
                        m_pca = m_model['pca'].transform(m_scaled)

                        # Fused Vector (11D + 34D = 45D)
                        fused_vec = np.hstack((m_pca, a_pca))

                    with st.spinner("Step 4: Mapping to Sonic Neighborhood..."):
                        fused_scaled = c_model['scaler'].transform(fused_vec)
                        centroids = c_model['centroids']
                        dist_to_centers = np.linalg.norm(centroids - fused_scaled, axis=1)
                        cluster_id = np.argmin(dist_to_centers)

                    with st.spinner("Step 5: Finding Similar Songs..."):
                        all_fused = np.stack(df['fused_features'].values)
                        norms_db = np.linalg.norm(all_fused, axis=1)
                        norm_q = np.linalg.norm(fused_vec)
                        
                        similarities = (all_fused @ fused_vec.T).flatten() / (norms_db * norm_q)
                        top_indices = np.argsort(similarities)[::-1][:5]
                        
                    st.session_state['discovery_results'] = {
                        'query': search_query,
                        'cluster_id': cluster_id,
                        'top_indices': top_indices,
                        'similarities': similarities
                    }

                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    
        if st.session_state.get('discovery_results') and st.session_state['discovery_results']['query'] == search_query:
            res = st.session_state['discovery_results']
            st.success(f"Analysis Complete! This song belongs to **Cluster {res['cluster_id']}**.")
            
            cluster_songs = df[df['cluster'] == res['cluster_id']]
            top_genres = cluster_songs['playlist_genre'].value_counts(normalize=True).head(3)
            genre_str = ", ".join([f"{g.upper()} ({p*100:.1f}%)" for g, p in top_genres.items()])
            st.info(f"📍 **Neighborhood Vibe:** Primarily {genre_str}")

            st.subheader("Similar Tracks in Database")
            cols = st.columns(5)
            for i, idx_match in enumerate(res['top_indices']):
                match_row = df.iloc[idx_match]
                with cols[i]:
                    st.markdown(f"""
                    <div class="song-card" style="height: 200px;">
                        <p style="font-weight: bold; margin-bottom: 5px;">{match_row['track_name']}</p>
                        <p style="color: #888; font-size: 0.8rem; margin-bottom: 10px;">{match_row['track_artist']}</p>
                        <span class="cluster-tag" style="background: #E2E8F0; color: #475569;">Sim: {res['similarities'][idx_match]*100:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    if st.button("Preview", key=f"disc_prev_{i}"):
                        play_song(match_row['track_name'], match_row['track_artist'])
                        st.rerun()


st.markdown("---")

# --- CLUSTER MAP (3D t-SNE) ---
st.subheader("Musical Landscape (3D Projection)")
st.markdown("Mapping **45D multimodal space** into 3D using t-SNE. Clusters represent shared sonic characteristics.")

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
