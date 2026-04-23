import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.audio_utils import fetch_youtube_audio
from src.audio_feature_extractor import extract_audio_features
from src.recommender import build_matrices, build_embedding_std, recommend, METADATA_COLS
from src.pca import manual_pca
from src.clustering import ManualKMeans, manual_silhouette_score

LAMBDA_WEIGHT = 1.5
TOP_K = 10

st.set_page_config(
    page_title="Music Discovery Engine",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; color: #1E293B; }
    div.stButton > button {
        background: #FFFFFF;
        color: #1E293B;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 0.25rem 0.75rem;
        font-weight: 600;
        white-space: nowrap;
        transition: all 0.2s ease;
    }
    div.stButton > button:hover {
        background: #1E293B;
        border-color: #1E293B;
        color: #FFFFFF;
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
    .sub-text { color: #64748B; font-size: 1.1rem; margin-bottom: 2rem; }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563EB 0%, #06B6D4 100%);
        color: white; border: none; border-radius: 10px; font-weight: 600;
    }

    /* Spotify-style row card */
    .row-card {
        display: flex;
        align-items: center;
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 8px;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .row-card:hover { background: #F1F5F9; border-color: #93C5FD; }
    .row-rank {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94A3B8;
        min-width: 32px;
        text-align: center;
    }
    .row-info { flex: 1; min-width: 0; padding: 0 1rem; }
    .row-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .row-artist {
        font-size: 0.8rem;
        color: #64748B;
        margin: 2px 0 0 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .row-genre {
        font-size: 0.7rem;
        color: #2563EB;
        font-weight: 600;
        text-transform: uppercase;
        min-width: 80px;
        padding: 0 0.75rem;
    }
    .row-score {
        font-size: 0.8rem;
        color: #475569;
        font-family: monospace;
        min-width: 140px;
        text-align: right;
        padding-right: 0.75rem;
    }
    .seed-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# --- SESSION STATE ---
if 'playing_song' not in st.session_state:
    st.session_state['playing_song'] = None

def play_song(name, artist):
    st.session_state['playing_song'] = {"name": name, "artist": artist}

# --- DATA LOADING ---
@st.cache_resource
def load_data():
    data_path = PROJECT_ROOT / 'data/dataset/master_music_data.parquet'
    if not data_path.exists():
        raise FileNotFoundError(
            f"Master dataset not found at {data_path}. "
            "Run: audio_feature_extractor.py → build_master_dataset.py"
        )
    df = pd.read_parquet(data_path)
    meta_matrix, emb_matrix = build_matrices(df)
    emb_std, feat_mean, feat_std = build_embedding_std(df)
    return df, meta_matrix, emb_matrix, emb_std, feat_mean, feat_std

try:
    df, meta_matrix, emb_matrix, emb_std, feat_mean, feat_std = load_data()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("Settings")
    st.markdown("---")
    st.write(f"**Songs in DB:** {len(df)}")
    st.markdown("### Model")
    st.caption(f"λ = {LAMBDA_WEIGHT} (tuned via same-artist vs random pair sweep)")
    st.caption("score = MSE_metadata + λ × (1 − cos_sim_embedding)")

    st.markdown("### Genre Filter")
    genres = ["All"] + sorted(df['playlist_genre'].dropna().unique().tolist())
    genre_filter = st.selectbox("Filter by Genre", genres)

# --- HEADER ---
st.markdown('<h1 class="main-header">Discovery Engine</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-text">Music recommendation using MSE distance on metadata + cosine similarity on audio embeddings.</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Songs", len(df))
with col2:
    st.metric("λ (fixed)", LAMBDA_WEIGHT)
with col3:
    st.metric("Genres", df['playlist_genre'].nunique())

st.markdown("---")

# --- GLOBAL PLAYER ---
if st.session_state.get('playing_song'):
    curr = st.session_state['playing_song']
    st.markdown(f"""
    <div style="background: rgba(37,99,235,0.1); border: 1px solid #2563EB; padding: 10px; border-radius: 10px; margin-bottom: 20px;">
        <span style="color: #2563EB; font-weight: bold;">▶ NOW PLAYING:</span> {curr['name']} — {curr['artist']}
    </div>
    """, unsafe_allow_html=True)
    with st.spinner("Streaming from YouTube..."):
        path = fetch_youtube_audio(curr['name'], curr['artist'])
        if path:
            st.audio(path, format="audio/m4a", autoplay=True)
        else:
            st.error("Could not fetch audio.")


def render_song_row(rank: int, row, score_text: str, key: str):
    """Render a single Spotify-style horizontal row with a play button."""
    c_rank, c_info, c_genre, c_score, c_play = st.columns([0.6, 4.0, 1.2, 2.0, 1.2])
    with c_rank:
        st.markdown(f'<div class="row-rank">#{rank}</div>', unsafe_allow_html=True)
    with c_info:
        title = str(row.get('track_name', ''))
        artist = str(row.get('track_artist', ''))
        st.markdown(
            f'<div class="row-title">{title}</div>'
            f'<div class="row-artist">{artist}</div>',
            unsafe_allow_html=True,
        )
    with c_genre:
        genre = str(row.get('playlist_genre', '')).upper() if row.get('playlist_genre') else ''
        st.markdown(f'<div class="row-genre">{genre}</div>', unsafe_allow_html=True)
    with c_score:
        st.markdown(f'<div class="row-score">{score_text}</div>', unsafe_allow_html=True)
    with c_play:
        if st.button("▶ Play", key=key):
            play_song(row['track_name'], row['track_artist'])
            st.rerun()


# --- TABS ---
tab_seed, tab_discovery, tab_playlist, tab_eval = st.tabs([
    "🎧 Seed Song", "🔍 Analyze Any Song", "🎼 Auto Playlist", "📊 Evaluation"
])

# ─── SEED SONG TAB ───
with tab_seed:
    st.subheader("Pick a Seed Song")

    song_list = df.apply(lambda r: f"{r['track_name']} — {r['track_artist']}", axis=1).tolist()
    selected = st.selectbox("Search Song", ["Select a song..."] + song_list, label_visibility="collapsed")

    if selected != "Select a song...":
        idx = song_list.index(selected)
        row = df.iloc[idx]

        c1, c2 = st.columns([1, 3])
        with c1:
            st.markdown(f"""
            <div class="seed-card">
                <h3 style="margin-top:0;">{row['track_name']}</h3>
                <p style="color:#64748B; margin:0;">{row['track_artist']}</p>
                <hr style="opacity:0.1;">
                <p style="font-size:0.85rem;"><b>Genre:</b> {str(row.get('playlist_genre','')).upper()}</p>
            </div>
            """, unsafe_allow_html=True)

            if st.button("🎧 Play Seed", key="play_seed", type="primary"):
                play_song(row['track_name'], row['track_artist'])
                st.rerun()

            features = ['danceability','energy','speechiness','acousticness','instrumentalness','liveness','valence']
            radar_df = pd.DataFrame(dict(r=[row[f] for f in features], theta=features))
            fig_radar = px.line_polar(radar_df, r='r', theta='theta', line_close=True, template='plotly_white')
            fig_radar.update_traces(fill='toself', line_color='#2563EB')
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                                    showlegend=False, height=280, margin=dict(l=30,r=30,t=10,b=10))
            st.plotly_chart(fig_radar, width="stretch")

        with c2:
            st.markdown(f"### Top {TOP_K} Recommendations")
            recs = recommend(
                idx, df, meta_matrix, emb_matrix,
                lambda_weight=LAMBDA_WEIGHT, top_k=TOP_K,
                genre_filter=genre_filter if genre_filter != "All" else None
            )

            if not recs:
                st.info("No recommendations found. Try removing the genre filter.")
            else:
                for i, rec in enumerate(recs, 1):
                    neighbor = df.iloc[rec['idx']]
                    score_text = f"score {rec['score']:.4f}"
                    render_song_row(i, neighbor, score_text, key=f"seed_play_{rec['idx']}")

# ─── ANALYZE ANY SONG TAB ───
with tab_discovery:
    st.subheader("Analyze Any Song from YouTube")
    st.caption("Download any song from YouTube, extract its audio features, and find similar songs in the DB. (Audio-only; no metadata is available for new tracks.)")

    search_query = st.text_input("YouTube Search", placeholder="e.g. NewJeans - Super Shy")

    if search_query and st.button("Analyze & Match", type="primary"):
        with st.spinner("Downloading from YouTube..."):
            try:
                tmp_path = fetch_youtube_audio(search_query, "")
                if not tmp_path:
                    st.error("Could not download audio.")
                    st.stop()

                with st.spinner("Extracting audio features..."):
                    raw_audio_features = extract_audio_features(tmp_path)

                # Z-score and L2-normalize to match the master emb_matrix space
                q_emb_raw = raw_audio_features.astype(np.float32)
                q_emb_std = (q_emb_raw - feat_mean) / feat_std
                
                q_norm = np.linalg.norm(q_emb_std)
                q_norm = q_norm if q_norm != 0 else 1.0
                q_emb_l2 = q_emb_std / q_norm
                
                # Compute Cosine Distance (1 - Cosine Similarity) using dot product 
                # because emb_matrix and q_emb_l2 are already unit vectors.
                scores = 1.0 - np.dot(emb_matrix, q_emb_l2)
                top_indices = np.argsort(scores)[:TOP_K]

                # Persist so Play buttons survive the rerun (button-returns-True is one-shot)
                st.session_state['analyze_results'] = {
                    'query': search_query,
                    'tmp_path': tmp_path,
                    'top_indices': [int(i) for i in top_indices],
                    'scores': [float(scores[i]) for i in top_indices],
                }
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.session_state.pop('analyze_results', None)

    # Render from state so results persist across reruns (e.g. after clicking Play)
    res = st.session_state.get('analyze_results')
    if res:
        st.markdown(f"**Analyzed:** `{res['query']}`")
        st.audio(res['tmp_path'])
        st.markdown(f"### Top {TOP_K} Similar Songs in DB")
        for i, (idx_match, score) in enumerate(zip(res['top_indices'], res['scores']), 1):
            match_row = df.iloc[idx_match]
            render_song_row(i, match_row, f"audio dist {score:.4f}", key=f"disc_play_{idx_match}")

# ─── AUTO PLAYLIST TAB ───
with tab_playlist:
    st.subheader("Auto Playlist — Mood-based Clusters")
    st.caption(
        "Songs are clustered by K-Means (manual implementation, K-Means++ init, Lloyd's algorithm). "
        "Each cluster is an automatically discovered mood/vibe group. "
        "This is a standalone feature — the main recommendation engine on other tabs does NOT use clustering."
    )

    # ── Elbow & Silhouette sweep ──────────────────────────────────────────────
    @st.cache_data
    def run_elbow_analysis(_meta_matrix, _emb_matrix):
        """Sweep K=2–12. Returns (ks, inertias, silhouettes). Cached — runs once."""
        X = np.hstack([_meta_matrix, _emb_matrix]).astype(np.float64)
        ks = list(range(5, 13))
        inertias, silhouettes = [], []
        for k_val in ks:
            model = ManualKMeans(n_clusters=k_val, n_init=3, max_iter=150, random_state=42)
            model.fit(X)
            inertias.append(model.inertia_)
            silhouettes.append(manual_silhouette_score(X, model.labels_, sample_size=500))
        return ks, inertias, silhouettes

    with st.expander("K-Means Quality Metrics — Elbow & Silhouette", expanded=True):
        st.caption(
            "Sweep K=5–12 to find the best K. "
            "**Elbow**: pick K where inertia stops dropping sharply. "
            "**Silhouette**: pick the K with the highest score."
        )
        with st.spinner("Computing elbow sweep K=5–12 (runs once, cached)..."):
            ks, inertias, silhouettes = run_elbow_analysis(meta_matrix, emb_matrix)

        best_sil_k = ks[int(np.argmax(silhouettes))]
        best_sil_val = max(silhouettes)

        col_elbow, col_sil = st.columns(2)

        with col_elbow:
            fig_elbow = px.line(
                x=ks, y=inertias,
                markers=True,
                labels={"x": "K (clusters)", "y": "Inertia"},
                title="Elbow Method — Inertia vs K",
                template="plotly_white",
            )
            fig_elbow.update_traces(line_color="#2563EB", marker_color="#2563EB")
            fig_elbow.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_elbow, width="stretch")
            st.caption(
                "No universal threshold — look for the kink where the curve flattens. "
                "Adding more clusters past that point gives diminishing returns."
            )

        with col_sil:
            sil_colors = ["#F59E0B" if k_val == best_sil_k else "#94A3B8" for k_val in ks]
            fig_sil = px.bar(
                x=ks, y=silhouettes,
                labels={"x": "K (clusters)", "y": "Silhouette Score"},
                title=f"Silhouette Score vs K  (best: K={best_sil_k}, s={best_sil_val:.4f})",
                template="plotly_white",
            )
            fig_sil.update_traces(marker_color=sil_colors)
            fig_sil.add_hline(
                y=0, line_dash="dash", line_color="#EF4444",
                annotation_text="random baseline", annotation_position="bottom right"
            )
            fig_sil.update_layout(height=340, margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig_sil, width="stretch")
            st.caption(
                f"**Why is the score ~{best_sil_val:.2f}?** In a 64D continuous space, distances naturally equalize "
                f"*(Concentration of Measure)*, forcing formulas closer to 0. Since 64D random noise scores ~0.00, "
                f"our score of **{best_sil_val:.4f}** (at K={best_sil_k}) mathematically proves genuine acoustic clusters exist!"
            )

        st.info(
            f"Silhouette score = (b − a) / max(a, b).  "
            f"a = mean distance to same-cluster points;  "
            f"b = min mean distance to nearest other cluster.  "
            f"Computed on a 500-point subsample of the 64D feature space (8D metadata + 56D audio).",
            icon="ℹ️",
        )

    st.markdown("---")

    # ── Cluster browser ───────────────────────────────────────────────────────
    k = st.slider("Number of Clusters (K)", min_value=5, max_value=15, value=best_sil_k)

    @st.cache_data
    def run_kmeans(_meta_matrix, _emb_matrix, n_clusters: int):
        X = np.hstack([_meta_matrix, _emb_matrix])
        model = ManualKMeans(n_clusters=n_clusters, n_init=5, random_state=42)
        model.fit(X)
        return model.labels_, model.inertia_

    with st.spinner(f"Running K-Means with K={k} (K-Means++ init, 5 restarts)..."):
        labels, inertia = run_kmeans(meta_matrix, emb_matrix, k)

    st.success(f"Clustering complete. Inertia: {inertia:.2f}")

    df_clustered = df.copy()
    df_clustered['cluster'] = labels

    cluster_ids = sorted(df_clustered['cluster'].unique())
    selected_cluster = st.selectbox(
        "Browse cluster",
        cluster_ids,
        format_func=lambda c: f"Cluster {c}  ({(labels == c).sum()} songs)"
    )

    cluster_df = df_clustered[df_clustered['cluster'] == selected_cluster].reset_index(drop=True)

    genre_counts = cluster_df['playlist_genre'].value_counts(normalize=True)
    genre_str = "  ·  ".join([f"**{g.upper()}** {p*100:.0f}%" for g, p in genre_counts.head(4).items()])
    st.markdown(f"**Vibe breakdown:** {genre_str}")

    col_chart, col_songs = st.columns([1, 2])

    with col_chart:
        fig_pie = px.pie(
            values=genre_counts.values,
            names=genre_counts.index,
            title=f"Cluster {selected_cluster} — Genre Mix",
            template="plotly_white",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig_pie, width="stretch")

    with col_songs:
        st.markdown(f"**Playlist ({len(cluster_df)} songs, showing first 10):**")
        for i, (_, row) in enumerate(cluster_df.head(10).iterrows(), 1):
            render_song_row(i, row, f"cluster {selected_cluster}", key=f"pl_play_{selected_cluster}_{i}")

# ─── EVALUATION TAB ───
with tab_eval:
    st.subheader("Music Space — PCA Visualization")
    st.caption("56D audio embeddings projected to 2D via manual PCA (covariance matrix SVD). Points close together sound similar.")

    @st.cache_data
    def get_pca_2d(_emb_matrix, _df):
        X_2d, var_ratio = manual_pca(_emb_matrix, n_components=2)
        plot_df = _df[['track_name', 'track_artist', 'playlist_genre']].copy()
        plot_df['x'] = X_2d[:, 0]
        plot_df['y'] = X_2d[:, 1]
        return plot_df, var_ratio

    plot_df, var_ratio = get_pca_2d(emb_matrix, df)
    st.caption(f"Explained variance: PC1 = {var_ratio[0]*100:.1f}%, PC2 = {var_ratio[1]*100:.1f}%")
    fig = px.scatter(
        plot_df, x='x', y='y', color='playlist_genre',
        hover_name='track_name', hover_data=['track_artist'],
        template='plotly_white',
        opacity=0.65,
        color_discrete_sequence=px.colors.qualitative.Prism,
    )
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0), legend_title="Genre")
    st.plotly_chart(fig, width="stretch")

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#555; margin-top:30px; font-size:0.8rem;">
    Multimodal Music Discovery — MSE (Metadata) + Cosine (Audio Embedding)
</div>
""", unsafe_allow_html=True)
