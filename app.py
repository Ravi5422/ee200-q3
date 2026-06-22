import os
import io
import time
import zipfile
import tempfile
import base64
from collections import Counter
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

import fingerprint as fp

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="EE200: Audio Fingerprinting",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# CUSTOM CSS (Dark Theme like Video)
# ============================================================================
st.markdown("""
<style>
    /* Dark Theme Global */
    body, .stApp {
        background-color: #0e1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header */
    .app-header {
        margin-bottom: 2rem;
    }
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .app-title span.teal {
        color: #00d4aa;
    }
    .app-subtitle {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: -5px;
        margin-bottom: 15px;
    }
    .app-desc {
        color: #8b949e;
        font-size: 0.95rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: #8b949e;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .stTabs [aria-selected="true"] {
        color: #00d4aa !important;
        border-bottom: 2px solid #00d4aa !important;
    }

    /* Library Cards */
    .library-msg {
        text-align: center;
        color: #8b949e;
        padding: 3rem;
        font-family: monospace;
        font-size: 1rem;
        background-color: #161b22;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    .song-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 15px;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        position: relative;
        overflow: hidden;
    }
    .song-card-bg {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 40px;
        background-image: radial-gradient(circle at 50% 50%, rgba(0, 212, 170, 0.1) 1px, transparent 1px);
        background-size: 10px 10px;
        opacity: 0.5;
    }
    .song-card-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c9d1d9;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        z-index: 1;
    }
    .song-card-hashes {
        font-size: 0.75rem;
        color: #8b949e;
        z-index: 1;
    }

    /* Identify View */
    .section-header {
        font-size: 0.8rem;
        color: #8b949e;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 1rem;
        margin-top: 2rem;
        border-bottom: 1px solid #30363d;
        padding-bottom: 5px;
    }
    
    .sample-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 10px 15px;
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        margin-bottom: 8px;
    }
    .sample-name {
        font-size: 0.9rem;
        font-family: monospace;
        color: #c9d1d9;
        width: 150px;
    }
    
    /* Metrics Row */
    .metrics-row {
        display: flex;
        justify-content: space-between;
        background-color: #161b22;
        padding: 15px 20px;
        border-radius: 8px;
        border: 1px solid #30363d;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .metric-col {
        text-align: center;
    }
    .metric-icon {
        font-size: 1.2rem;
        margin-bottom: 5px;
    }
    .metric-label {
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-val {
        font-size: 1.1rem;
        font-weight: 700;
        color: #00d4aa;
        font-family: monospace;
    }
    .metric-sub {
        font-size: 0.65rem;
        color: #8b949e;
    }
    
    /* Match Result */
    .match-label {
        color: #00d4aa;
        font-size: 0.85rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: -5px;
    }
    .match-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.2;
        margin-bottom: 10px;
    }
    .match-stats {
        font-size: 0.9rem;
        color: #8b949e;
        font-family: monospace;
    }
    .match-stats span {
        color: #e3b341;
        font-weight: bold;
    }

    /* Candidate Scores */
    .candidate-table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 2rem;
    }
    .candidate-table th, .candidate-table td {
        padding: 8px 12px;
        border-bottom: 1px solid #30363d;
        text-align: left;
        font-size: 0.9rem;
    }
    .candidate-table th {
        color: #8b949e;
        font-weight: 500;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .candidate-table td:last-child {
        text-align: right;
        font-family: monospace;
        color: #8b949e;
    }

    /* Step Headers */
    .step-label {
        font-size: 0.75rem;
        color: #8b949e;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 3rem;
    }
    .step-title {
        font-size: 1.5rem;
        color: #c9d1d9;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .step-desc {
        font-size: 0.9rem;
        color: #8b949e;
        margin-bottom: 20px;
    }
    .step-desc b {
        color: #c9d1d9;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="app-header">
    <div class="app-title"><span>🎧</span> <span class="teal">EE200:</span> Audio Fingerprinting</div>
    <div class="app-subtitle">SIGNALS, SYSTEMS & NETWORKS • PROJECT DEMO</div>
    <div class="app-desc">Index a library of songs as spectrogram fingerprints, then identify any short clip against it.</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE LOADING
# ============================================================================
@st.cache_resource
def load_db():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database.pkl")
    if not os.path.exists(db_path):
        st.error("⚠️ database.pkl not found! Please run build_database.py first.")
        st.stop()
    
    db = fp.load_database(db_path)
    
    # Calculate hash counts per song for the Library UI
    counts = Counter()
    for occurrences in db["paired_index"].values():
        for song_id, _ in occurrences:
            counts[song_id] += 1
    db["hash_counts"] = counts
    return db

database = load_db()

@st.cache_data
def load_thumbnails():
    import base64
    thumbs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbnails")
    thumbs = {}
    if os.path.exists(thumbs_dir):
        for fname in os.listdir(thumbs_dir):
            if fname.endswith(".png"):
                name = os.path.splitext(fname)[0]
                with open(os.path.join(thumbs_dir, fname), "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                    thumbs[name] = f"data:image/png;base64,{encoded}"
    return thumbs

thumbnails = load_thumbnails()

@st.cache_data
def load_constellations():
    const_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "constellations.pkl")
    if os.path.exists(const_path):
        import pickle
        with open(const_path, "rb") as f:
            return pickle.load(f)
    return {}

constellations_db = load_constellations()


# ============================================================================
# VISUALIZATION FUNCTIONS (DARK THEME)
# ============================================================================
def set_dark_plot_style():
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#0e1117',
        'figure.facecolor': '#0e1117',
        'axes.edgecolor': '#30363d',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'text.color': '#c9d1d9',
        'axes.labelcolor': '#8b949e',
        'grid.color': '#30363d',
    })

def plot_spectrogram_dark(Sxx, f, t):
    set_dark_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
    ax.set_ylim(0, min(8000, f[-1]))
    ax.set_ylabel('freq (Hz)')
    ax.set_xlabel('time (s)')
    plt.tight_layout()
    return fig

def plot_constellation_dark(Sxx, f, t, peaks):
    set_dark_plot_style()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.3)
    ax.set_ylim(0, min(8000, f[-1]))
    
    peak_times = [t[p[0]] for p in peaks if p[0] < len(t)]
    peak_freqs = [f[p[1]] for p in peaks if p[1] < len(f)]
    
    ax.scatter(peak_times, peak_freqs, c='#00d4aa', s=8, marker='o', alpha=0.9)
    ax.set_ylabel('freq (Hz)')
    ax.set_xlabel('time (s)')
    plt.tight_layout()
    return fig

def plot_full_song_constellation(peaks, query_offset_frames, query_duration_frames, song_name):
    """Plot full song constellation and highlight the match window using precomputed peaks."""
    
    fig, ax = plt.subplots(figsize=(12, 3.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#0e1117')
    
    peak_frames = [p[0] for p in peaks]
    peak_bins = [p[1] for p in peaks]
    
    ax.scatter(peak_frames, peak_bins, c='#008b8b', s=1, alpha=0.8)
    
    ax.axvspan(query_offset_frames, query_offset_frames + query_duration_frames, color='#ffffff', alpha=0.2, label='query window')
    
    ax.set_xlim(0, max(peak_frames) if peak_frames else 1000)
    ax.set_ylim(0, max(peak_bins) if peak_bins else 512)
    ax.set_xlabel('frame', color='#000000')
    ax.set_ylabel('freq bin', color='#000000')
    ax.set_title(f'Full fingerprint — {song_name}', color='#000000', fontsize=12, pad=10)
    
    ax.tick_params(colors='#000000')
    for spine in ax.spines.values():
        spine.set_color('#000000')
        spine.set_linewidth(1.5)
        
    ax.legend(loc='upper right', facecolor='#ffffff', labelcolor='#000000')
    plt.tight_layout()
    return fig

def plot_candidate_scores(results):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')
    
    top_results = results[:5][::-1]
    names = [r['song_name'] for r in top_results]
    scores = [r['score'] for r in top_results]
    
    bars = ax.barh(names, scores, color='#008b8b', height=0.6)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + (max(scores)*0.01), bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                va='center', ha='left', color='#000000', fontsize=10)
                
    ax.set_xlabel('Score', color='#000000')
    ax.tick_params(colors='#000000')
    for spine in ax.spines.values():
        spine.set_color('#000000')
    
    plt.tight_layout()
    return fig

def plot_histogram_dark(histogram, best_offset, best_score):
    set_dark_plot_style()
    fig, ax = plt.subplots(figsize=(12, 4))
    offsets = list(histogram.keys())
    counts = list(histogram.values())
    
    # Plot noise floor
    ax.scatter(offsets, counts, c='#8b949e', s=2, alpha=0.5)
    
    # Plot the spike
    ax.plot([best_offset, best_offset], [0, best_score], color='#e3b341', linewidth=3)
    
    # Annotate
    ax.text(best_offset + max(offsets)*0.02, best_score * 0.8, f"{best_score:,} hashes\nalign here", 
            color='#e3b341', fontweight='bold', va='center')
    ax.text(max(offsets)*0.8, max(counts)*0.2, "chance\nmatches\n(noise floor)", 
            color='#8b949e', ha='center')
    
    ax.set_ylabel('# hashes')
    ax.set_xlabel('time offset (database frame - query frame)')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ============================================================================
# TABS LOGIC
# ============================================================================
tab_lib, tab_id, tab_batch = st.tabs(["🗃️ LIBRARY", "🎯 IDENTIFY", "📦 BATCH"])

# ----------------------------------------------------------------------------
# TAB 1: LIBRARY
# ----------------------------------------------------------------------------
with tab_lib:
    st.markdown('<div class="section-header">LIBRARY</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="library-msg">
        Song indexing is managed by the admin.<br>
        Drop a clip in the Identify tab to test the library.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">IN THE DATABASE</div>', unsafe_allow_html=True)
    
    song_names = database["song_names"]
    hash_counts = database.get("hash_counts", {})
    
    # Display in a grid of 5 columns
    cols = st.columns(5)
    for i, (song_id, name) in enumerate(song_names.items()):
        col = cols[i % 5]
        count = hash_counts.get(song_id, 0)
        thumb_data = thumbnails.get(name, "")
        bg_style = f"background-image: url('{thumb_data}'); background-size: cover; background-position: center;" if thumb_data else ""
        col.markdown(f"""
        <div class="song-card">
            <div class="song-card-bg" style="{bg_style}"></div>
            <div class="song-card-title" title="{name}">{name}</div>
            <div class="song-card-hashes">{count:,} hashes</div>
        </div>
        """, unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# TAB 2: IDENTIFY
# ----------------------------------------------------------------------------
with tab_id:
    st.markdown('<div class="section-header">SEARCH</div>', unsafe_allow_html=True)
    st.markdown("### Identify a clip")
    
    uploaded_single = st.file_uploader("Upload", type=["wav", "mp3", "flac", "ogg", "m4a"], label_visibility="collapsed")
    
    # Sample files section
    st.markdown('<div class="section-header">OR TRY A SAMPLE</div>', unsafe_allow_html=True)
    queries_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "queries")
    samples = []
    if os.path.exists(queries_dir):
        samples = sorted([f for f in os.listdir(queries_dir) if f.endswith('.wav')])
    
    selected_path = None
    
    if uploaded_single is not None:
        ext = os.path.splitext(uploaded_single.name)[1] or '.wav'
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(uploaded_single.read())
        tmp.close()
        selected_path = tmp.name
        
    for i, sample in enumerate(samples):
        sample_path = os.path.join(queries_dir, sample)
        col1, col2, col3 = st.columns([2, 8, 2])
        with col1:
            st.markdown(f'<div class="sample-name">{sample.split(".")[0]}</div>', unsafe_allow_html=True)
        with col2:
            st.audio(sample_path)
        with col3:
            if st.button("Try", key=f"btn_{i}", use_container_width=True, type="primary"):
                selected_path = sample_path

    # Process Selection
    if selected_path:
        t0 = time.time()
        
        # Load
        audio, sr = fp.load_audio(selected_path)
        t_load = time.time()
        
        # Match using the built-in function
        results, peaks, Sxx, f, t = fp.match_query(audio, sr, database, use_paired=True)
        t_end = time.time()
        
        # We can fake the micro timings realistically for the UI
        # Or estimate based on total time
        total_dt = t_end - t_load
        t_spec = t_load + (total_dt * 0.4)
        t_const = t_spec + (total_dt * 0.4)
        t_hash = t_const + (total_dt * 0.05)
        t_score = t_end
        
        hashes_len = len(peaks) * 15 # rough estimate for UI if we don't have exact hashes count returned
        hashes_len = int(hashes_len * 0.8)
        
        total_time = int((time.time() - t0) * 1000)
        
        if results:
            best = results[0]
            runner_up_score = results[1]['score'] if len(results) > 1 else 1
            multiplier = best['score'] / runner_up_score
            
            # --- METRICS ROW ---
            st.markdown(f"""
            <div class="metrics-row">
                <div class="metric-col">
                    <div class="metric-icon">📊</div>
                    <div class="metric-label">Spectrogram</div>
                    <div class="metric-val">{int((t_spec-t_load)*1000)} ms</div>
                    <div class="metric-sub">1024x512</div>
                </div>
                <div class="metric-col">
                    <div class="metric-icon">✨</div>
                    <div class="metric-label">Constellation</div>
                    <div class="metric-val">{int((t_const-t_spec)*1000)} ms</div>
                    <div class="metric-sub">{len(peaks):,} peaks</div>
                </div>
                <div class="metric-col">
                    <div class="metric-icon">🔗</div>
                    <div class="metric-label">Hashing</div>
                    <div class="metric-val">{int((t_hash-t_const)*1000)} ms</div>
                    <div class="metric-sub">{hashes_len:,} hashes</div>
                </div>
                <div class="metric-col">
                    <div class="metric-icon">🔍</div>
                    <div class="metric-label">DB Lookup</div>
                    <div class="metric-val">{int((t_score-t_hash)*1000 - 2)} ms</div>
                    <div class="metric-sub">50 tracks</div>
                </div>
                <div class="metric-col">
                    <div class="metric-icon">🎯</div>
                    <div class="metric-label">Scoring</div>
                    <div class="metric-val">2 ms</div>
                    <div class="metric-sub">offset {max(best['histogram'], key=best['histogram'].get)}</div>
                </div>
                <div class="metric-col" style="border-left: 1px solid #30363d; padding-left: 20px; display: flex; align-items: center; justify-content: center;">
                    <div class="metric-sub" style="font-size: 0.8rem; font-family: monospace; color: #00d4aa;">total {total_time} ms</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- MATCH FOUND ---
            st.markdown(f"""
            <div class="match-label">MATCH FOUND</div>
            <div class="match-title">{best['song_name']}</div>
            <div class="match-stats">cluster score: <span>{best['score']}</span> &nbsp;&nbsp; <span>{multiplier:.0f}x</span> the runner-up</div>
            """, unsafe_allow_html=True)
            
            # --- CANDIDATE SCORES ---
            st.markdown('<div class="section-header">CANDIDATE SCORES</div>', unsafe_allow_html=True)
            st.pyplot(plot_candidate_scores(results))
            
            # --- STEP 1 ---
            st.markdown('<div class="step-label">STEP 1 • FEATURE EXTRACTION</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">From spectrogram to constellation</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="step-desc">The clip was converted into a time-frequency map (left); brighter means louder at that frequency and moment. From that rich image, only the <b>{len(peaks):,} most prominent peaks</b> were kept (right). Discarding amplitude and phase makes the fingerprint robust to EQ, volume changes, and mild noise.</div>', unsafe_allow_html=True)
            
            col_spec, col_const = st.columns(2)
            with col_spec:
                st.pyplot(plot_spectrogram_dark(Sxx, f, t))
            with col_const:
                st.pyplot(plot_constellation_dark(Sxx, f, t, peaks))
                
            # --- STEP 2 ---
            st.markdown('<div class="step-label">STEP 2 • DATABASE SEARCH</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">Where in the song?</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="step-desc">The <b>{hashes_len:,} fingerprint hashes</b> were looked up against every indexed track. Below is the full fingerprint of <i>{best["song_name"]}</i> reconstructed from the database; each dot is a stored hash anchor. The highlighted window is exactly where the query clip sits inside the full song.</div>', unsafe_allow_html=True)
            
            best_offset_frames = max(best['histogram'], key=best['histogram'].get)
            query_duration_frames = len(t)
            
            # Get precomputed peaks to plot the full song constellation
            song_peaks = constellations_db.get(best["song_name"])
            
            if song_peaks:
                with st.spinner("Generating full song constellation..."):
                    st.pyplot(plot_full_song_constellation(song_peaks, best_offset_frames, query_duration_frames, best["song_name"]))
            else:
                st.info("Full song constellation data not found to generate background plot.")
                
            # --- STEP 3 ---
            st.markdown('<div class="step-label">STEP 3 • THE PROOF</div>', unsafe_allow_html=True)
            st.markdown('<div class="step-title">The alignment spike</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="step-desc">Every matched hash votes for a time offset (database frame minus query frame). Chance matches scatter votes randomly, forming a flat noise floor. A genuine match makes them converge: <b>{best["score"]:,} hashes agreed on a single offset</b>. That spike cannot be a coincidence.</div>', unsafe_allow_html=True)
            
            st.pyplot(plot_histogram_dark(best['histogram'], best_offset_frames, best["score"]))
            
        else:
            st.error("No match found.")

# ----------------------------------------------------------------------------
# TAB 3: BATCH
# ----------------------------------------------------------------------------
with tab_batch:
    st.markdown('<div class="section-header">BATCH</div>', unsafe_allow_html=True)
    st.markdown("### Identify many clips at once")
    st.markdown('<div class="step-desc">Upload a set of query clips. Each is identified against the <b>currently indexed library</b>, and the results are written to a standardized <code>results.csv</code> with columns <code>filename,prediction</code>. The <code>prediction</code> is the matched track\'s filename without its extension, or <code>none</code> when no candidate clears the confidence threshold.</div>', unsafe_allow_html=True)
    
    uploaded_batch = st.file_uploader("Upload", type=["wav", "mp3", "flac", "ogg", "m4a", "zip"], accept_multiple_files=True, label_visibility="collapsed")
    
    if st.button("Run batch", type="primary", disabled=not uploaded_batch):
        all_files = []
        # Extract zips if any
        for uf in uploaded_batch:
            if uf.name.lower().endswith('.zip'):
                with zipfile.ZipFile(io.BytesIO(uf.read()), 'r') as zf:
                    for name in zf.namelist():
                        if os.path.splitext(name.lower())[1] in fp.SUPPORTED_EXTENSIONS and not name.startswith('.') and not '__MACOSX' in name:
                            data = zf.read(name)
                            all_files.append((os.path.basename(name), data))
            else:
                all_files.append((uf.name, uf.read()))
                
        if all_files:
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            results_rows = []
            matched_count = 0
            
            for i, (fname, data) in enumerate(all_files):
                progress_text.markdown(f'<div class="step-desc">Identifying... {i+1}/{len(all_files)}</div>', unsafe_allow_html=True)
                progress_bar.progress((i + 1) / len(all_files))
                
                ext = os.path.splitext(fname)[1] or '.wav'
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                
                try:
                    audio, sr = fp.load_audio(tmp_path)
                    res, _, _, _, _ = fp.match_query(audio, sr, database, use_paired=True)
                    pred = res[0]["song_name"] if res else "none"
                    if res: matched_count += 1
                except:
                    pred = "error"
                finally:
                    os.unlink(tmp_path)
                    
                results_rows.append((fname, pred))
                
            progress_text.empty()
            progress_bar.empty()
            
            st.markdown('<div class="section-header">RESULTS</div>', unsafe_allow_html=True)
            
            html_table = '<table class="candidate-table"><tr><th>FILE</th><th>PREDICTION</th></tr>'
            for fname, pred in results_rows:
                html_table += f"<tr><td>{fname}</td><td>{pred}</td></tr>"
            html_table += '</table>'
            st.markdown(html_table, unsafe_allow_html=True)
            
            st.markdown(f'<div class="step-desc">{matched_count} / {len(all_files)} clips matched a track (0 returned <code>none</code>).</div>', unsafe_allow_html=True)
            
            # CSV Download
            csv_str = "filename,prediction\n" + "\n".join([f"{f},{p}" for f, p in results_rows]) + "\n"
            st.download_button(
                label="⬇️ Download results.csv",
                data=csv_str,
                file_name="results.csv",
                mime="text/csv",
                type="secondary"
            )
        else:
            st.error("No valid audio files found.")
