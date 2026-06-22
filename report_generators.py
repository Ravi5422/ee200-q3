"""
report_generators.py — Experiment Figures for the EE200 Q3 Report
=================================================================
Generates all six experiment plots required for the PDF report.
Each figure is saved to ./figures/ at 300 DPI for print quality.

Usage:
    python report_generators.py --songs_dir ./songs

Experiments:
    1. DFT vs Spectrogram
    2. Window Size Trade-off
    3. Constellation Plot
    4. Single vs Paired Hash Histograms
    5. Robustness vs Noise (Accuracy vs SNR)
    6. Pitch Shift & Time Stretch Analysis
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless servers
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Add project root to path so we can import fingerprint module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fingerprint as fp


# ---------------------------------------------------------------------------
# Helper: pick the first song in a directory
# ---------------------------------------------------------------------------
def _get_first_song(songs_dir):
    """Return the path to the first audio file found in songs_dir."""
    audio_files = sorted([f for f in os.listdir(songs_dir)
                          if os.path.splitext(f.lower())[1] in fp.SUPPORTED_EXTENSIONS])
    if not audio_files:
        raise FileNotFoundError(f"No audio files in {songs_dir}")
    return os.path.join(songs_dir, audio_files[0]), audio_files[0]


def _get_second_song(songs_dir):
    """Return the path to the second audio file (or first if only one)."""
    audio_files = sorted([f for f in os.listdir(songs_dir)
                          if os.path.splitext(f.lower())[1] in fp.SUPPORTED_EXTENSIONS])
    if len(audio_files) < 2:
        return os.path.join(songs_dir, audio_files[0]), audio_files[0]
    return os.path.join(songs_dir, audio_files[1]), audio_files[1]


# ============================================================================
# EXPERIMENT 1: DFT vs Spectrogram
# ============================================================================

def experiment_1_dft_vs_spectrogram(songs_dir, figures_dir):
    """
    Plot the 1D DFT magnitude of an entire song alongside its spectrogram.

    The DFT shows *which* frequencies are present but loses all timing info —
    you cannot tell *when* each frequency occurred.  The spectrogram preserves
    both time and frequency.
    """
    filepath, fname = _get_first_song(songs_dir)
    audio, sr = fp.load_audio(filepath)
    song_name = os.path.splitext(fname)[0]

    # --- 1D DFT of the whole signal -----------------------------------------
    N = len(audio)
    fft_vals = np.fft.rfft(audio)
    fft_mag = np.abs(fft_vals)
    fft_freqs = np.fft.rfftfreq(N, d=1.0/sr)

    # --- Spectrogram ---------------------------------------------------------
    f, t, Sxx = fp.compute_spectrogram(audio, sr, nperseg=1024, noverlap=512)

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Top: 1D DFT
    axes[0].plot(fft_freqs, fft_mag, color='steelblue', linewidth=0.3)
    axes[0].set_xlim(0, sr / 2)
    axes[0].set_xlabel('Frequency (Hz)', fontsize=12)
    axes[0].set_ylabel('Magnitude', fontsize=12)
    axes[0].set_title(f'1D DFT Magnitude — "{song_name}"\n'
                      '(All frequency content collapsed — no timing information)',
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Bottom: Spectrogram
    im = axes[1].pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
    axes[1].set_ylim(0, 8000)  # focus on audible range
    axes[1].set_xlabel('Time (s)', fontsize=12)
    axes[1].set_ylabel('Frequency (Hz)', fontsize=12)
    axes[1].set_title(f'Spectrogram — "{song_name}"\n'
                      '(Frequency content preserved across time)',
                      fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=axes[1], label='Power (dB)')

    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'dft_vs_spectrogram.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")


# ============================================================================
# EXPERIMENT 2: Window Size Trade-off
# ============================================================================

def experiment_2_window_tradeoff(songs_dir, figures_dir):
    """
    Show the time-frequency resolution trade-off by plotting two spectrograms
    of the same clip with different window sizes.

    Short window (nperseg=256):  good time resolution, poor frequency resolution
    Long  window (nperseg=4096): good frequency resolution, poor time resolution
    """
    filepath, fname = _get_first_song(songs_dir)
    audio, sr = fp.load_audio(filepath)
    song_name = os.path.splitext(fname)[0]

    # Use first 10 seconds for clarity
    clip = fp.extract_clip(audio, sr, start_sec=0, duration_sec=10)

    # Two different window sizes
    configs = [
        (256,  128,  'Short Window (nperseg=256)\nGood time resolution, poor frequency resolution'),
        (4096, 2048, 'Long Window (nperseg=4096)\nGood frequency resolution, poor time resolution'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (nperseg, noverlap, title) in zip(axes, configs):
        f, t, Sxx = fp.compute_spectrogram(clip, sr, nperseg=nperseg,
                                            noverlap=noverlap)
        im = ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma')
        ax.set_ylim(0, 8000)
        ax.set_xlabel('Time (s)', fontsize=11)
        ax.set_ylabel('Frequency (Hz)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        fig.colorbar(im, ax=ax, label='Power (dB)')

    fig.suptitle(f'Window Size Trade-off — "{song_name}" (first 10 s)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'window_tradeoff.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")


# ============================================================================
# EXPERIMENT 3: Constellation Plot
# ============================================================================

def experiment_3_constellation(songs_dir, figures_dir):
    """
    Plot a spectrogram with the extracted constellation (local maxima)
    overlaid as scatter points.
    """
    filepath, fname = _get_first_song(songs_dir)
    audio, sr = fp.load_audio(filepath)
    song_name = os.path.splitext(fname)[0]

    # Use first 15 seconds
    clip = fp.extract_clip(audio, sr, start_sec=0, duration_sec=15)
    f, t, Sxx = fp.compute_spectrogram(clip, sr, nperseg=1024, noverlap=512)
    peaks = fp.find_peaks(Sxx, neighborhood_size=20, threshold_factor=2.0)

    # Convert peak indices to physical units for plotting
    peak_times = [t[p[0]] for p in peaks if p[0] < len(t)]
    peak_freqs = [f[p[1]] for p in peaks if p[1] < len(f)]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Spectrogram background
    ax.pcolormesh(t, f, Sxx, shading='gouraud', cmap='magma', alpha=0.8)
    ax.set_ylim(0, 8000)

    # Constellation overlay
    ax.scatter(peak_times, peak_freqs, c='cyan', s=8, marker='o',
               edgecolors='white', linewidths=0.3, alpha=0.9,
               label=f'Constellation peaks ({len(peaks):,} points)')

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Frequency (Hz)', fontsize=12)
    ax.set_title(f'Spectrogram with Constellation — "{song_name}" (first 15 s)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.8)

    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'constellation.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")


# ============================================================================
# EXPERIMENT 4: Single vs Paired Hash Histograms
# ============================================================================

def experiment_4_single_vs_paired(songs_dir, figures_dir):
    """
    Run matching on a noisy query clip using (a) single-peak hashes and
    (b) paired hashes.  Plot the offset histograms side-by-side.

    The paired histogram should show a clear, decisive spike at the correct
    offset, while the single-peak histogram is noisy and ambiguous.
    """
    # Build a small in-memory database
    db = fp.build_database(songs_dir, db_path='__temp_exp4.pkl', verbose=False)

    # Pick a song and extract a clip with moderate noise
    filepath, fname = _get_first_song(songs_dir)
    audio, sr = fp.load_audio(filepath)
    song_name = os.path.splitext(fname)[0]
    clip = fp.extract_clip(audio, sr, start_sec=5, duration_sec=8)
    noisy_clip = fp.add_noise(clip, snr_db=15)  # moderate noise

    # --- Match with PAIRED hashes ---
    paired_results, _, _, _, _ = fp.match_query(
        noisy_clip, sr, db, use_paired=True
    )

    # --- Match with SINGLE hashes ---
    single_results, _, _, _, _ = fp.match_query(
        noisy_clip, sr, db, use_paired=False
    )

    # --- Plot histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Single hashes histogram (top match)
    if single_results:
        best_single = single_results[0]
        hist_single = best_single["histogram"]
        offsets_s = list(hist_single.keys())
        counts_s = list(hist_single.values())
        axes[0].bar(offsets_s, counts_s, width=1.0, color='salmon', alpha=0.7)
        axes[0].set_title(
            f'Single-Peak Hashes\n'
            f'Best match: "{best_single["song_name"]}" '
            f'(score={best_single["score"]})',
            fontsize=12, fontweight='bold'
        )
    else:
        axes[0].set_title('Single-Peak Hashes\nNo matches found',
                          fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Time Offset Δ (bins)', fontsize=11)
    axes[0].set_ylabel('Number of Aligned Hashes', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Paired hashes histogram (top match)
    if paired_results:
        best_paired = paired_results[0]
        hist_paired = best_paired["histogram"]
        offsets_p = list(hist_paired.keys())
        counts_p = list(hist_paired.values())
        axes[1].bar(offsets_p, counts_p, width=1.0, color='mediumseagreen',
                    alpha=0.7)
        axes[1].set_title(
            f'Paired Hashes\n'
            f'Best match: "{best_paired["song_name"]}" '
            f'(score={best_paired["score"]})',
            fontsize=12, fontweight='bold'
        )
    else:
        axes[1].set_title('Paired Hashes\nNo matches found',
                          fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Time Offset Δ (bins)', fontsize=11)
    axes[1].set_ylabel('Number of Aligned Hashes', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f'Offset Histograms — Query from "{song_name}" (SNR=15 dB)\n'
        f'Paired hashing produces a decisive spike; single peaks are ambiguous.',
        fontsize=13, fontweight='bold', y=1.04
    )
    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'single_vs_paired_histogram.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")

    # Clean up temporary database
    if os.path.exists('__temp_exp4.pkl'):
        os.remove('__temp_exp4.pkl')


# ============================================================================
# EXPERIMENT 5: Robustness vs Noise  (Accuracy vs SNR)
# ============================================================================

def experiment_5_noise_robustness(songs_dir, figures_dir):
    """
    Add increasing Gaussian noise to query clips and test whether the
    matcher still identifies the correct song.

    Plots Accuracy (%) vs SNR (dB).  We test several songs, extract clips
    from each, and average the accuracy.
    """
    db = fp.build_database(songs_dir, db_path='__temp_exp5.pkl', verbose=False)

    wav_files = sorted([
        f for f in os.listdir(songs_dir)
        if os.path.splitext(f.lower())[1] in fp.SUPPORTED_EXTENSIONS
    ])

    # SNR levels to test (high → low noise)
    snr_levels = [40, 30, 25, 20, 15, 10, 5, 3, 1, 0, -3, -5]

    accuracy_per_snr = []

    for snr_db in snr_levels:
        correct = 0
        total = 0

        for fname in wav_files:
            filepath = os.path.join(songs_dir, fname)
            audio, sr = fp.load_audio(filepath)
            song_name = os.path.splitext(fname)[0]

            # Extract a 5-second clip from the middle of the song
            duration = len(audio) / sr
            start = max(0, duration / 2 - 2.5)
            clip = fp.extract_clip(audio, sr, start_sec=start, duration_sec=5)

            # Add noise
            noisy_clip = fp.add_noise(clip, snr_db=snr_db)

            # Match
            results, _, _, _, _ = fp.match_query(noisy_clip, sr, db,
                                                  use_paired=True)
            if results and results[0]["song_name"] == song_name:
                correct += 1
            total += 1

        accuracy = (correct / total * 100) if total > 0 else 0
        accuracy_per_snr.append(accuracy)
        print(f"    SNR={snr_db:+3d} dB  →  Accuracy={accuracy:.1f}% "
              f"({correct}/{total})")

    # --- Plot ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(snr_levels, accuracy_per_snr, 'o-', color='royalblue',
            linewidth=2, markersize=8, markerfacecolor='white',
            markeredgecolor='royalblue', markeredgewidth=2)
    ax.set_xlabel('SNR (dB)', fontsize=13)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Robustness to Noise — Accuracy vs SNR',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.4, label='50% baseline')
    ax.legend(fontsize=11)

    # Annotate each point
    for snr, acc in zip(snr_levels, accuracy_per_snr):
        ax.annotate(f'{acc:.0f}%', (snr, acc), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=9)

    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'accuracy_vs_snr.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")

    # Clean up
    if os.path.exists('__temp_exp5.pkl'):
        os.remove('__temp_exp5.pkl')


# ============================================================================
# EXPERIMENT 6: Pitch Shift & Time Stretch
# ============================================================================

def experiment_6_pitch_time(songs_dir, figures_dir):
    """
    Apply slight pitch shifts and time stretches to a query clip and test
    whether the matcher still identifies the correct song.

    Explains *why* pitch shifting breaks the match: it shifts all spectral
    peaks to different frequency bins, so the hash tuples (f1, f2, Δt)
    no longer match the database.  Similarly, time-stretching changes the
    Δt component.
    """
    try:
        import librosa
    except ImportError:
        print("  ⚠ librosa not installed — skipping Experiment 6.")
        print("    Install with: pip install librosa")
        return

    db = fp.build_database(songs_dir, db_path='__temp_exp6.pkl', verbose=False)

    filepath, fname = _get_first_song(songs_dir)
    audio, sr = fp.load_audio(filepath)
    song_name = os.path.splitext(fname)[0]

    # Extract a clean 5-second clip
    duration = len(audio) / sr
    start = max(0, duration / 2 - 2.5)
    clip = fp.extract_clip(audio, sr, start_sec=start, duration_sec=5)

    # --- Test various pitch shifts ---
    pitch_shifts = [-4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4]
    pitch_results = []

    print("    Pitch shift tests:")
    for semitones in pitch_shifts:
        shifted = librosa.effects.pitch_shift(clip, sr=sr, n_steps=semitones)
        results, _, _, _, _ = fp.match_query(shifted, sr, db, use_paired=True)
        matched = (results and results[0]["song_name"] == song_name)
        score = results[0]["score"] if results else 0
        pitch_results.append({
            "semitones": semitones,
            "matched": matched,
            "score": score,
        })
        status = "✓" if matched else "✗"
        print(f"      {status}  shift={semitones:+.1f} semitones  "
              f"score={score}")

    # --- Test various time stretches ---
    time_rates = [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]
    time_results = []

    print("    Time stretch tests:")
    for rate in time_rates:
        stretched = librosa.effects.time_stretch(clip, rate=rate)
        results, _, _, _, _ = fp.match_query(stretched, sr, db, use_paired=True)
        matched = (results and results[0]["song_name"] == song_name)
        score = results[0]["score"] if results else 0
        time_results.append({
            "rate": rate,
            "matched": matched,
            "score": score,
        })
        status = "✓" if matched else "✗"
        print(f"      {status}  rate={rate:.2f}x  score={score}")

    # --- Plot ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Pitch shift plot
    semitones_list = [r["semitones"] for r in pitch_results]
    scores_pitch = [r["score"] for r in pitch_results]
    colors_pitch = ['mediumseagreen' if r["matched"] else 'salmon'
                    for r in pitch_results]
    axes[0].bar(semitones_list, scores_pitch, width=0.5, color=colors_pitch,
                edgecolor='gray', alpha=0.8)
    axes[0].set_xlabel('Pitch Shift (semitones)', fontsize=12)
    axes[0].set_ylabel('Match Score', fontsize=12)
    axes[0].set_title(
        'Pitch Shift Robustness\n'
        'Green = correct match, Red = failed\n'
        '(Shifting moves freq bins → hashes change)',
        fontsize=12, fontweight='bold'
    )
    axes[0].grid(True, alpha=0.3, axis='y')

    # Time stretch plot
    rates_list = [r["rate"] for r in time_results]
    scores_time = [r["score"] for r in time_results]
    colors_time = ['mediumseagreen' if r["matched"] else 'salmon'
                   for r in time_results]
    axes[1].bar([str(r) for r in rates_list], scores_time, color=colors_time,
                edgecolor='gray', alpha=0.8)
    axes[1].set_xlabel('Time Stretch Rate', fontsize=12)
    axes[1].set_ylabel('Match Score', fontsize=12)
    axes[1].set_title(
        'Time Stretch Robustness\n'
        'Green = correct match, Red = failed\n'
        '(Stretching changes Δt in hash → mismatch)',
        fontsize=12, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    fig.suptitle(f'Pitch & Time Robustness — "{song_name}"',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    outpath = os.path.join(figures_dir, 'pitch_time_analysis.png')
    fig.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {outpath}")

    # --- Also save a text summary for the report ---
    summary_path = os.path.join(figures_dir, 'pitch_time_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("PITCH SHIFT & TIME STRETCH ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write("WHY PITCH SHIFTING BREAKS MATCHING:\n")
        f.write("  A pitch shift of n semitones multiplies all frequencies\n")
        f.write("  by 2^(n/12).  This moves every spectral peak to a\n")
        f.write("  different frequency bin.  Since the paired hash is\n")
        f.write("  (f1, f2, Δt), both f1 and f2 change, producing\n")
        f.write("  entirely different hash keys that don't exist in\n")
        f.write("  the database.\n\n")
        f.write("WHY TIME STRETCHING BREAKS MATCHING:\n")
        f.write("  Time stretching by rate r maps time t → t/r.\n")
        f.write("  The Δt component of each hash (f1, f2, Δt) changes,\n")
        f.write("  so the hash keys no longer match even though the\n")
        f.write("  frequency components f1, f2 are preserved.\n\n")
        f.write("RESULTS:\n")
        f.write("-" * 50 + "\n")
        f.write("Pitch Shifts:\n")
        for r in pitch_results:
            status = "PASS" if r["matched"] else "FAIL"
            f.write(f"  {r['semitones']:+.1f} semitones: {status} "
                    f"(score={r['score']})\n")
        f.write("\nTime Stretches:\n")
        for r in time_results:
            status = "PASS" if r["matched"] else "FAIL"
            f.write(f"  {r['rate']:.2f}x: {status} (score={r['score']})\n")
    print(f"  ✓ Saved: {summary_path}")

    # Clean up
    if os.path.exists('__temp_exp6.pkl'):
        os.remove('__temp_exp6.pkl')


# ============================================================================
# MAIN — run all experiments
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate all experiment figures for the EE200 Q3 report.'
    )
    parser.add_argument(
        '--songs_dir', type=str, default='./songs',
        help='Directory containing audio files (default: ./songs)'
    )
    parser.add_argument(
        '--figures_dir', type=str, default='./figures',
        help='Directory to save figures (default: ./figures)'
    )
    parser.add_argument(
        '--experiments', type=str, default='all',
        help='Comma-separated list of experiment numbers to run, '
             'e.g. "1,3,5" or "all" (default: all)'
    )
    args = parser.parse_args()

    # Create figures directory
    os.makedirs(args.figures_dir, exist_ok=True)

    # Determine which experiments to run
    if args.experiments.lower() == 'all':
        exps_to_run = [1, 2, 3, 4, 5, 6]
    else:
        exps_to_run = [int(x.strip()) for x in args.experiments.split(',')]

    experiment_funcs = {
        1: ("DFT vs Spectrogram",           experiment_1_dft_vs_spectrogram),
        2: ("Window Size Trade-off",        experiment_2_window_tradeoff),
        3: ("Constellation Plot",           experiment_3_constellation),
        4: ("Single vs Paired Histograms",  experiment_4_single_vs_paired),
        5: ("Noise Robustness (Acc vs SNR)",experiment_5_noise_robustness),
        6: ("Pitch Shift & Time Stretch",   experiment_6_pitch_time),
    }

    print("=" * 60)
    print("  EE200 Q3 — Report Figure Generator")
    print("=" * 60)
    print(f"  Songs directory : {args.songs_dir}")
    print(f"  Figures output  : {args.figures_dir}")
    print(f"  Experiments     : {exps_to_run}")
    print("=" * 60)

    for exp_num in exps_to_run:
        if exp_num not in experiment_funcs:
            print(f"\n⚠ Unknown experiment number: {exp_num}")
            continue
        name, func = experiment_funcs[exp_num]
        print(f"\n▶ Experiment {exp_num}: {name}")
        try:
            func(args.songs_dir, args.figures_dir)
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("  All experiments complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
