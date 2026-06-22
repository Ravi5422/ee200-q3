"""
fingerprint.py — Core Audio Fingerprinting Engine
===================================================
Implements a Shazam-style audio fingerprinting system from scratch using
only numpy and scipy.  No external fingerprinting libraries are used.

Pipeline overview:
    1. Load audio  →  mono waveform at native sample-rate
    2. Compute STFT spectrogram  (scipy.signal.spectrogram)
    3. Extract constellation points  (local-maxima via scipy.ndimage.maximum_filter)
    4. Build combinatorial hashes  (paired peaks) or simple hashes (single peaks)
    5. Store hashes in an inverted-index database  (dict  →  pickle)
    6. Match a query clip by histogram-of-offsets voting

Author : EE200 — Q3 Sonic Signatures & Zapptain America
"""

import os
import pickle
import numpy as np
import librosa
from scipy.signal import spectrogram as scipy_spectrogram
from scipy.ndimage import maximum_filter
from collections import defaultdict, Counter

# Supported audio file extensions (add more if needed)
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}


# ============================================================================
# 1. AUDIO LOADING
# ============================================================================

def load_audio(filepath, sr=None):
    """
    Load an audio file (WAV, MP3, FLAC, etc.) and return (audio_mono, sample_rate).

    - Uses librosa.load() which handles MP3, WAV, and many other formats.
    - Automatically converts stereo → mono.
    - Returns float32 normalised to [-1, 1].

    Parameters
    ----------
    filepath : str
        Path to an audio file (.wav, .mp3, .flac, etc.).
    sr : int or None
        Target sample rate.  None = use the file's native rate.

    Returns
    -------
    audio : np.ndarray, shape (N,)
        Mono audio signal normalised to [-1, 1].
    sr : int
        Sample rate in Hz.
    """
    # librosa.load returns (audio_float32, sample_rate)
    # mono=True averages channels, sr=None keeps native rate
    audio, sr = librosa.load(filepath, sr=sr, mono=True)
    audio = audio.astype(np.float64)
    return audio, sr


# ============================================================================
# 2. SPECTROGRAM COMPUTATION
# ============================================================================

def compute_spectrogram(audio, sr, nperseg=1024, noverlap=512):
    """
    Compute the Short-Time Fourier Transform (STFT) spectrogram.

    Uses a Hann window and returns the *log-magnitude* spectrogram
    which compresses the dynamic range and makes peaks easier to detect.

    Parameters
    ----------
    audio : np.ndarray
        Mono audio signal.
    sr : int
        Sample rate.
    nperseg : int
        Number of samples per STFT segment (window length).
        Controls the frequency resolution: longer → sharper frequency,
        but blurrier time.
    noverlap : int
        Number of overlapping samples between consecutive segments.

    Returns
    -------
    f : np.ndarray, shape (n_freq,)
        Frequency axis in Hz.
    t : np.ndarray, shape (n_time,)
        Time axis in seconds.
    Sxx : np.ndarray, shape (n_freq, n_time)
        Log-magnitude spectrogram (dB-like).
    """
    f, t, Sxx = scipy_spectrogram(
        audio,
        fs=sr,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Convert power spectrogram to log scale (add small epsilon to avoid log(0))
    Sxx = 10 * np.log10(Sxx + 1e-10)

    return f, t, Sxx


# ============================================================================
# 3. CONSTELLATION EXTRACTION  (peak picking)
# ============================================================================

def find_peaks(Sxx, neighborhood_size=20, threshold_factor=2.0):
    """
    Extract a sparse "constellation" of time-frequency peaks from the
    spectrogram.

    A pixel (f_bin, t_bin) is kept if:
        1. It equals the local maximum in a (neighborhood_size × neighborhood_size)
           region  (scipy.ndimage.maximum_filter).
        2. Its value exceeds  mean + threshold_factor × std  of the whole
           spectrogram  (rejects quiet / noisy regions).

    Parameters
    ----------
    Sxx : np.ndarray, shape (n_freq, n_time)
        Log-magnitude spectrogram.
    neighborhood_size : int
        Side length of the square neighbourhood used for local-max filtering.
    threshold_factor : float
        How many standard deviations above the mean a peak must be.

    Returns
    -------
    peaks : list of (time_bin, freq_bin)
        Sparse constellation points, sorted by time_bin.
    """
    # Step 1: Find local maxima — each pixel that equals the max of its
    #         neighbourhood passes through unchanged; others are suppressed.
    local_max = maximum_filter(Sxx, size=neighborhood_size)
    is_local_max = (Sxx == local_max)

    # Step 2: Apply an amplitude threshold to reject weak peaks / noise floor
    threshold = Sxx.mean() + threshold_factor * Sxx.std()
    is_above_threshold = (Sxx > threshold)

    # Combine both conditions
    detected = is_local_max & is_above_threshold

    # Extract (freq_bin, time_bin) indices where peaks were found
    freq_bins, time_bins = np.where(detected)

    # Return as list of (time_bin, freq_bin) — time first for chronological sort
    peaks = list(zip(time_bins.tolist(), freq_bins.tolist()))
    peaks.sort(key=lambda p: p[0])  # sort by time

    return peaks


# ============================================================================
# 4a. PAIRED HASHING  (the good approach)
# ============================================================================

def generate_paired_hashes(peaks, fan_value=15, delta_time_max=200):
    """
    Create combinatorial (paired) hashes from constellation points.

    For each "anchor" peak, we pair it with up to `fan_value` subsequent
    peaks that lie within `delta_time_max` time-bins ahead.  Each pair
    produces a hash:

        hash = (freq_anchor, freq_target, delta_time)

    mapped to:

        value = anchor_time_bin   (absolute position in the song)

    This combinatorial approach is much more discriminative than hashing
    individual peaks because it captures *relative timing* between nearby
    spectral events.

    Parameters
    ----------
    peaks : list of (time_bin, freq_bin)
        Constellation points (sorted by time).
    fan_value : int
        Max number of future peaks to pair with each anchor.
    delta_time_max : int
        Maximum allowed time difference (in bins) for a pair.

    Returns
    -------
    hashes : list of (hash_value, anchor_time)
        hash_value is a tuple (f1, f2, dt).
        anchor_time is the absolute time-bin of the anchor peak.
    """
    hashes = []
    n = len(peaks)

    for i in range(n):
        anchor_time, anchor_freq = peaks[i]
        # Look at the next `fan_value` peaks (or fewer near the end)
        paired = 0
        for j in range(i + 1, n):
            target_time, target_freq = peaks[j]
            dt = target_time - anchor_time

            # Skip if the target is too far in the future
            if dt > delta_time_max:
                break

            # Skip if delta-time is zero (same time slot — unlikely but possible)
            if dt <= 0:
                continue

            # Create the hash tuple
            hash_val = (anchor_freq, target_freq, dt)
            hashes.append((hash_val, anchor_time))

            paired += 1
            if paired >= fan_value:
                break

    return hashes


# ============================================================================
# 4b. SINGLE-PEAK HASHING  (the naïve approach, for comparison)
# ============================================================================

def generate_single_hashes(peaks):
    """
    Create simple single-peak hashes.  Each peak produces a hash
    consisting of its frequency bin alone:

        hash = (freq_bin,)
        value = time_bin

    This is much less discriminative because many different songs share the
    same frequency peaks.  It is implemented purely to demonstrate *why*
    paired hashing is superior (see Experiment 4).

    Parameters
    ----------
    peaks : list of (time_bin, freq_bin)
        Constellation points.

    Returns
    -------
    hashes : list of (hash_value, time_bin)
    """
    hashes = []
    for t_bin, f_bin in peaks:
        hash_val = (f_bin,)
        hashes.append((hash_val, t_bin))
    return hashes


# ============================================================================
# 5. DATABASE BUILDING  (inverted index)
# ============================================================================

def build_database(songs_dir, db_path="database.pkl",
                   nperseg=1024, noverlap=512,
                   neighborhood_size=20, threshold_factor=2.0,
                   fan_value=15, delta_time_max=200,
                   verbose=True):
    """
    Index all audio files in `songs_dir` and write a fingerprint database.

    Supports .wav, .mp3, .flac, .ogg, .m4a and other formats via librosa.

    The database is a Python dict serialised with pickle:
        {
            "paired_index":  {hash_tuple: [(song_id, offset), ...], ...},
            "single_index":  {hash_tuple: [(song_id, offset), ...], ...},
            "song_names":    {song_id: "filename_without_ext", ...},
            "params": {                    # reproducibility record
                "nperseg": ..., "noverlap": ..., ...
            }
        }

    Parameters
    ----------
    songs_dir : str
        Directory containing audio files (.wav, .mp3, etc.).
    db_path : str
        Where to save the pickle database.
    (remaining params)
        Forwarded to spectrogram / peak / hash functions.
    verbose : bool
        Print progress.

    Returns
    -------
    database : dict
        The in-memory database (also saved to disk).
    """
    paired_index = defaultdict(list)   # hash → [(song_id, offset), ...]
    single_index = defaultdict(list)
    song_names = {}

    # Collect all supported audio files (case-insensitive)
    audio_files = sorted([
        f for f in os.listdir(songs_dir)
        if os.path.splitext(f.lower())[1] in SUPPORTED_EXTENSIONS
    ])

    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in '{songs_dir}'. "
            "Supported formats: " + ", ".join(SUPPORTED_EXTENSIONS)
        )

    for song_id, fname in enumerate(audio_files):
        song_name = os.path.splitext(fname)[0]   # ground-truth label
        song_names[song_id] = song_name
        filepath = os.path.join(songs_dir, fname)

        if verbose:
            print(f"  [{song_id+1}/{len(audio_files)}] Indexing: {fname}")

        # Load → spectrogram → peaks → hashes
        audio, sr = load_audio(filepath)
        _, _, Sxx = compute_spectrogram(audio, sr, nperseg, noverlap)
        peaks = find_peaks(Sxx, neighborhood_size, threshold_factor)

        # Paired hashes
        p_hashes = generate_paired_hashes(peaks, fan_value, delta_time_max)
        for h, offset in p_hashes:
            paired_index[h].append((song_id, offset))

        # Single hashes (for experiment comparison)
        s_hashes = generate_single_hashes(peaks)
        for h, offset in s_hashes:
            single_index[h].append((song_id, offset))

        if verbose:
            print(f"         peaks={len(peaks):,}  "
                  f"paired_hashes={len(p_hashes):,}  "
                  f"single_hashes={len(s_hashes):,}")

    # Bundle into a single database dict
    database = {
        "paired_index": dict(paired_index),
        "single_index": dict(single_index),
        "song_names":   song_names,
        "params": {
            "nperseg": nperseg,
            "noverlap": noverlap,
            "neighborhood_size": neighborhood_size,
            "threshold_factor": threshold_factor,
            "fan_value": fan_value,
            "delta_time_max": delta_time_max,
        },
    }

    # Save to disk
    with open(db_path, "wb") as f:
        pickle.dump(database, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        total_paired = sum(len(v) for v in paired_index.values())
        total_single = sum(len(v) for v in single_index.values())
        print(f"\n✓ Database saved to '{db_path}'")
        print(f"  Songs indexed   : {len(song_names)}")
        print(f"  Unique paired   : {len(paired_index):,} hashes  "
              f"({total_paired:,} entries)")
        print(f"  Unique single   : {len(single_index):,} hashes  "
              f"({total_single:,} entries)")

    return database


def load_database(db_path="database.pkl"):
    """
    Load a previously-built fingerprint database from disk.

    Parameters
    ----------
    db_path : str
        Path to the pickle database.

    Returns
    -------
    database : dict
    """
    with open(db_path, "rb") as f:
        database = pickle.load(f)
    return database


# ============================================================================
# 6. MATCHING ENGINE
# ============================================================================

def match_query(query_audio, sr, database, use_paired=True,
                nperseg=1024, noverlap=512,
                neighborhood_size=20, threshold_factor=2.0,
                fan_value=15, delta_time_max=200,
                top_n=5):
    """
    Identify a query audio clip against the fingerprint database.

    Algorithm:
        1. Compute spectrogram & constellation of the query.
        2. Generate hashes (paired or single).
        3. For every hash that hits the DB, record (song_id, offset_diff)
           where offset_diff = db_offset - query_offset.
        4. For each song, build a histogram of offset_diffs.
           A true match produces a sharp spike (many hashes align at the
           same offset), while false matches produce scattered noise.
        5. Return the song whose histogram peak is tallest.

    Parameters
    ----------
    query_audio : np.ndarray
        Mono audio of the query clip.
    sr : int
        Sample rate.
    database : dict
        The fingerprint database (from build_database / load_database).
    use_paired : bool
        If True, use paired hashes (recommended).
        If False, use single-peak hashes (for experiment comparison).
    top_n : int
        Number of top candidates to return.

    Returns
    -------
    results : list of dict
        Top-N matches sorted by confidence (descending).  Each dict has:
            "song_id", "song_name", "score" (peak height),
            "offset" (best-aligned offset), "histogram" (full Counter).
    query_peaks : list of (time_bin, freq_bin)
        The constellation points of the query (for visualisation).
    query_Sxx : np.ndarray
        The query spectrogram (for visualisation).
    query_f : np.ndarray
        Frequency axis.
    query_t : np.ndarray
        Time axis.
    """
    # Read params from the database (use stored values for consistency)
    params = database.get("params", {})
    nperseg = params.get("nperseg", nperseg)
    noverlap = params.get("noverlap", noverlap)
    neighborhood_size = params.get("neighborhood_size", neighborhood_size)
    threshold_factor = params.get("threshold_factor", threshold_factor)
    fan_value = params.get("fan_value", fan_value)
    delta_time_max = params.get("delta_time_max", delta_time_max)

    # --- Step 1 & 2: spectrogram → peaks → hashes ---------------------------
    query_f, query_t, query_Sxx = compute_spectrogram(
        query_audio, sr, nperseg, noverlap
    )
    query_peaks = find_peaks(query_Sxx, neighborhood_size, threshold_factor)

    if use_paired:
        query_hashes = generate_paired_hashes(
            query_peaks, fan_value, delta_time_max
        )
        index = database["paired_index"]
    else:
        query_hashes = generate_single_hashes(query_peaks)
        index = database["single_index"]

    # --- Step 3: Look up each query hash in the DB ---------------------------
    # offset_counts[song_id] is a Counter of offset differences
    offset_counts = defaultdict(Counter)

    for h, query_offset in query_hashes:
        if h in index:
            for song_id, db_offset in index[h]:
                diff = db_offset - query_offset
                offset_counts[song_id][diff] += 1

    # --- Step 4: Find the tallest histogram spike per song -------------------
    results = []
    song_names = database["song_names"]

    for song_id, histogram in offset_counts.items():
        if not histogram:
            continue
        best_offset, score = histogram.most_common(1)[0]
        results.append({
            "song_id":   song_id,
            "song_name": song_names.get(song_id, f"unknown_{song_id}"),
            "score":     score,
            "offset":    best_offset,
            "histogram": histogram,
        })

    # Sort by score (highest first)
    results.sort(key=lambda r: r["score"], reverse=True)

    return results[:top_n], query_peaks, query_Sxx, query_f, query_t


# ============================================================================
# 7. UTILITY HELPERS
# ============================================================================

def add_noise(audio, snr_db):
    """
    Add white Gaussian noise to an audio signal at a given SNR (in dB).

    SNR = 10 * log10(P_signal / P_noise)

    Parameters
    ----------
    audio : np.ndarray
        Clean audio signal.
    snr_db : float
        Desired signal-to-noise ratio in decibels.

    Returns
    -------
    noisy : np.ndarray
        Audio with additive Gaussian noise.
    """
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.randn(len(audio)) * np.sqrt(noise_power)
    return audio + noise


def extract_clip(audio, sr, start_sec, duration_sec):
    """
    Extract a short clip from a longer audio signal.

    Parameters
    ----------
    audio : np.ndarray
        Full audio signal.
    sr : int
        Sample rate.
    start_sec : float
        Start time in seconds.
    duration_sec : float
        Duration in seconds.

    Returns
    -------
    clip : np.ndarray
    """
    start = int(start_sec * sr)
    end = int((start_sec + duration_sec) * sr)
    end = min(end, len(audio))
    return audio[start:end]
