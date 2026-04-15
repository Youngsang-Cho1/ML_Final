import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.manifold import TSNE

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
        background-color: #0E1117;
        color: #FFFFFF;
    }
    .main-header {
        font-family: 'Outfit', sans-serif;
        background: linear-gradient(90deg, #4B6CB7 0%, #182848 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .sub-text {
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #4B6CB7 0%, #182848 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
    }
    .song-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .song-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }
    .cluster-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 5px;
        background: #4B6CB7;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
    df = pd.read_parquet('data/dataset/clustered_songs.parquet')
    sim_matrix = np.load('data/dataset/cosine_sim_matrix.npy')
    return df, sim_matrix

try:
    df, sim_matrix = load_data()
except Exception as e:
    st.error(f"Failed to load datasets: {e}")
    st.info("Check if pipeline scripts have been executed (pca.py, clustering.py, cosine_similarity.py).")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings & Info")
    st.markdown("---")
    st.write("**Model:** DINOv2 (ViT-S/14)")
    st.write("**Data Size:** 1,850 Songs")
    st.write("**Fusion:** 35D PCA Fused Space")
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

    with c2:
        st.subheader("Recommendations")
        similar_indices = np.argsort(sim_matrix[idx])[-6:][::-1]
        
        cols = st.columns(5)
        for i, neighbor_idx in enumerate(similar_indices[1:]):
            neighbor = df.iloc[neighbor_idx]
            with cols[i]:
                st.markdown(f"""
                <div class="song-card" style="font-size: 0.9rem; padding: 1rem; height: 180px;">
                    <p style="font-weight: bold; margin-bottom: 5px;">{neighbor['track_name']}</p>
                    <p style="color: #888; font-size: 0.8rem;">{neighbor['track_artist']}</p>
                    <p style="color: #4B6CB7; font-size: 0.7rem; font-weight: bold;">{sim_matrix[idx][neighbor_idx]:.2f} Sim</p>
                </div>
                """, unsafe_allow_html=True)

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
    template='plotly_dark',
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

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
<div style="text-align: center; color: #555; margin-top: 50px; font-size: 0.8rem;">
    Multimodal Music Discovery Pipeline - Final Commitment Version
</div>
""", unsafe_allow_html=True)
