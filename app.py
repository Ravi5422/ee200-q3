"""
app.py — Streamlit Web Application for Audio Fingerprinting
============================================================
A two-mode GUI for the "Sonic Signatures & Zapptain America" system (EE200 Q3).

Mode 1: Single-Clip Identification
    Upload one audio clip → see prediction + spectrogram + constellation + histogram

Mode 2: Batch Processing (Strict Grading Format)
    Upload multiple audio clips (or a .zip) → download results.csv
    CSV format: exactly two columns "filename,prediction" with no index

Usage:
    streamlit run app.py

The app loads the pre-built database.pkl on startup, so it works immediately.
"""

import os
import io
import sys
import zipfile
import tempfile
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Add project root to path so we can import fingerprint module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fingerprint as fp


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sonic Signatures — Audio Fingerprinting",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# CUSTOM CSS — make it look professional
# ============================================================================
st.markdown("""
<style>
    /* Main header styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 0.5rem 0;
    }
    .sub-title {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #333;
    }
    .result-label {
        color: #aaa;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    .result-song {
        color: #00d4aa;
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .result-score {
        color: #667eea;
        font-size: 1.1rem;
    }
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
    }
    [data-testid="stSidebar"] .stMarkdown {
        color: #ddd;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# DATABASE LOADING (cached so it only loads once)
# ============================================================================

@st.cache_resource
def load_db():
    """Load the pre-built fingerprint database."""
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "database.pkl")
    if not os.path.exists(db_path):
        st.error(
            "⚠️ **database.pkl not found!** "
            "Please run `python build_database.py --songs_dir ./songs` first."
        )
        st.stop()
    return fp.load_database(db_path)


# ============================================================================
# VISUALISATION HELPERS
# ============================================================================

def plot_spectrogram(Sxx, f, t, title="Spectrogram"):
    """Plot a log-magnitude spectrogram with a beautiful colour map."""
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
    ax.set_ylim(0, min(8000, f[-1]))
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=ax, label='Power (dB)', shrink=0.8)
    plt.tight_layout()
    return fig


def plot_constellation(Sxx, f, t, peaks, title="Constellation Map"):
    """Spectrogram with constellation peaks overlaid."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.8)
    ax.set_ylim(0, min(8000, f[-1]))

    # Convert peak indices to physical units
    peak_times = [t[p[0]] for p in peaks if p[0] < len(t)]
    peak_freqs = [f[p[1]] for p in peaks if p[1] < len(f)]

    ax.scatter(peak_times, peak_freqs, c='cyan', s=6, marker='o',
               edgecolors='white', linewidths=0.2, alpha=0.9,
               label=f'{len(peaks):,} peaks')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Frequency (Hz)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.7)
    plt.tight_layout()
    return fig


def plot_histogram(histogram, song_name, score, title="Offset Histogram"):
    """Plot the offset histogram for the best match."""
    fig, ax = plt.subplots(figsize=(12, 4))
    offsets = list(histogram.keys())
    counts = list(histogram.values())

    ax.bar(offsets, counts, width=1.0, color='#667eea', alpha=0.7,
           edgecolor='none')

    # Highlight the peak
    best_offset = max(histogram, key=histogram.get)
    ax.axvline(x=best_offset, color='#00d4aa', linestyle='--', linewidth=2,
               label=f'Peak @ offset={best_offset} (score={score})')

    ax.set_xlabel('Time Offset Δ (bins)', fontsize=11)
    ax.set_ylabel('Number of Aligned Hashes', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    return fig


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_single_file(uploaded_file, database):
    """Process a single uploaded audio file and return results + visuals."""
    # Save uploaded file to a temporary location (keep original extension
    # so librosa can detect the codec — important for .mp3)
    ext = os.path.splitext(uploaded_file.name)[1] or '.wav'
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Load audio
        audio, sr = fp.load_audio(tmp_path)

        # Run matching
        results, peaks, Sxx, f_axis, t_axis = fp.match_query(
            audio, sr, database, use_paired=True
        )

        return {
            "results": results,
            "peaks": peaks,
            "Sxx": Sxx,
            "f": f_axis,
            "t": t_axis,
            "audio": audio,
            "sr": sr,
        }
    finally:
        os.unlink(tmp_path)


def process_batch_files(uploaded_files, database):
    """Process multiple files and return a list of (filename, prediction)."""
    rows = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        ext = os.path.splitext(uploaded_file.name)[1] or '.wav'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            audio, sr = fp.load_audio(tmp_path)
            results, _, _, _, _ = fp.match_query(
                audio, sr, database, use_paired=True
            )
            prediction = results[0]["song_name"] if results else "unknown"
        except Exception as e:
            prediction = "error"
        finally:
            os.unlink(tmp_path)

        rows.append((filename, prediction))

    return rows


def process_zip_file(zip_file, database):
    """Extract audio files from a .zip and process them all."""
    rows = []

    with tempfile.TemporaryDirectory() as tmpdir:
        # Extract zip
        with zipfile.ZipFile(io.BytesIO(zip_file.read()), 'r') as zf:
            zf.extractall(tmpdir)

        # Find all audio files (may be nested in subdirectories)
        audio_files = []
        for root, dirs, files in os.walk(tmpdir):
            for fname in sorted(files):
                ext = os.path.splitext(fname.lower())[1]
                if ext in fp.SUPPORTED_EXTENSIONS and not fname.startswith('.'):
                    audio_files.append((fname, os.path.join(root, fname)))

        for fname, fpath in audio_files:
            try:
                audio, sr = fp.load_audio(fpath)
                results, _, _, _, _ = fp.match_query(
                    audio, sr, database, use_paired=True
                )
                prediction = results[0]["song_name"] if results else "unknown"
            except Exception as e:
                prediction = "error"

            rows.append((fname, prediction))

    return rows


def rows_to_csv(rows):
    """Convert list of (filename, prediction) to CSV string (strict format)."""
    lines = ["filename,prediction"]
    for fname, pred in rows:
        lines.append(f"{fname},{pred}")
    return "\n".join(lines) + "\n"


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("## 🎵 Sonic Signatures")
    st.markdown("**Zapptain America**")
    st.markdown("---")

    mode = st.radio(
        "Select Mode",
        ["🔍 Single-Clip Identification", "📦 Batch Processing"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### How It Works")
    st.markdown("""
    1. **Spectrogram** — STFT of the audio
    2. **Constellation** — Local maxima peaks
    3. **Paired Hashes** — (f₁, f₂, Δt) tuples
    4. **Histogram Voting** — Offset alignment
    """)

    st.markdown("---")
    st.markdown(
        "<small>EE200 Course Project — Q3</small>",
        unsafe_allow_html=True
    )


# ============================================================================
# MAIN CONTENT
# ============================================================================

# Header
st.markdown('<div class="main-title">🎵 Sonic Signatures</div>',
            unsafe_allow_html=True)
st.markdown('<div class="sub-title">Audio Fingerprinting Engine — '
            'Identify any song from a short clip</div>',
            unsafe_allow_html=True)

# Load database
database = load_db()
n_songs = len(database.get("song_names", {}))
st.info(f"📂 Database loaded: **{n_songs} songs** indexed")


# --------------------------------------------------------------------------
# MODE 1: Single-Clip Identification
# --------------------------------------------------------------------------
if "Single" in mode:
    st.markdown("## 🔍 Single-Clip Identification")
    st.markdown("Upload an audio clip (`.wav`, `.mp3`, etc.) to identify the song.")

    uploaded = st.file_uploader(
        "Upload a query clip",
        type=["wav", "mp3", "flac", "ogg", "m4a"],
        key="single_upload",
    )

    if uploaded is not None:
        with st.spinner("🎧 Analyzing audio fingerprint..."):
            result = process_single_file(uploaded, database)

        results = result["results"]

        if results:
            best = results[0]

            # --- Display the main result ---
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Identified Song</div>
                <div class="result-song">🎶 {best['song_name']}</div>
                <div class="result-score">
                    Confidence Score: {best['score']} aligned hashes
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # --- Show top candidates if there are alternatives ---
            if len(results) > 1:
                with st.expander("🏆 Top Candidates", expanded=False):
                    for i, r in enumerate(results):
                        medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i] \
                            if i < 5 else f"{i+1}."
                        st.markdown(
                            f"{medal} **{r['song_name']}** — "
                            f"score: {r['score']}"
                        )

            # --- Visualisations ---
            st.markdown("---")
            st.markdown("### 📊 Analysis Pipeline Visualisations")

            # 1. Spectrogram
            st.markdown("#### 1. Query Spectrogram")
            fig_spec = plot_spectrogram(
                result["Sxx"], result["f"], result["t"],
                title=f"Query Spectrogram — \"{uploaded.name}\""
            )
            st.pyplot(fig_spec)
            plt.close(fig_spec)

            # 2. Constellation
            st.markdown("#### 2. Constellation of Peaks")
            fig_const = plot_constellation(
                result["Sxx"], result["f"], result["t"], result["peaks"],
                title=f"Constellation — {len(result['peaks']):,} peaks extracted"
            )
            st.pyplot(fig_const)
            plt.close(fig_const)

            # 3. Offset Histogram
            st.markdown("#### 3. Offset Histogram (Match Evidence)")
            fig_hist = plot_histogram(
                best["histogram"], best["song_name"], best["score"],
                title=f"Offset Histogram — Best match: \"{best['song_name']}\""
            )
            st.pyplot(fig_hist)
            plt.close(fig_hist)

        else:
            st.warning("⚠️ No match found. The clip may be too short or "
                       "too noisy, or the song is not in the database.")


# --------------------------------------------------------------------------
# MODE 2: Batch Processing
# --------------------------------------------------------------------------
else:
    st.markdown("## 📦 Batch Processing")
    st.markdown(
        "Upload audio clips (`.wav`, `.mp3`, etc.) or a `.zip` archive. "
        "Download results as `results.csv`."
    )

    # File uploader — accept audio formats and zip
    uploaded_files = st.file_uploader(
        "Upload query clips or a .zip archive",
        type=["wav", "mp3", "flac", "ogg", "m4a", "zip"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    if uploaded_files:
        # Separate zip files from wav files
        zip_files = [f for f in uploaded_files if f.name.lower().endswith('.zip')]
        wav_files = [f for f in uploaded_files
                     if os.path.splitext(f.name.lower())[1] in fp.SUPPORTED_EXTENSIONS]

        all_rows = []

        # Process zip files first
        if zip_files:
            for zf in zip_files:
                with st.spinner(f"📦 Extracting & processing {zf.name}..."):
                    rows = process_zip_file(zf, database)
                    all_rows.extend(rows)

        # Process individual wav files
        if wav_files:
            with st.spinner(f"🎧 Processing {len(wav_files)} clips..."):
                rows = process_batch_files(wav_files, database)
                all_rows.extend(rows)

        if all_rows:
            # Display results table
            st.markdown(f"### ✅ Results — {len(all_rows)} clips processed")

            # Show as a nice table
            import pandas as pd
            df = pd.DataFrame(all_rows, columns=["filename", "prediction"])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Generate CSV (STRICT FORMAT: filename,prediction — no index)
            csv_content = rows_to_csv(all_rows)

            # Download button
            st.download_button(
                label="⬇️ Download results.csv",
                data=csv_content,
                file_name="results.csv",
                mime="text/csv",
            )

            # Show CSV preview
            with st.expander("📄 CSV Preview (strict format)", expanded=False):
                st.code(csv_content, language="csv")
        else:
            st.warning("No audio files found in the uploaded content.")
