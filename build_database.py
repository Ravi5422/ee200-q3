#!/usr/bin/env python3
"""
build_database.py
=================
Command-line utility for building an audio-fingerprint database from a
directory of .wav song files.

Usage examples
--------------
    # Use default paths (./songs → ./database.pkl)
    python build_database.py

    # Specify a custom songs directory and output file
    python build_database.py --songs_dir /path/to/music --output /path/to/db.pkl

The heavy lifting is delegated to `fingerprint.build_database()`, which
reads every .wav file in the given directory, computes spectral peaks,
generates combinatorial hashes, and persists the resulting lookup table
as a pickle file.
"""

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import argparse   # For parsing command-line arguments
import os         # For filesystem path checks
import sys        # For controlled exit on errors
import time       # For timing the database-build process

# ---------------------------------------------------------------------------
# Local project import – the fingerprinting engine lives in fingerprint.py
# at the same directory level as this script.
# ---------------------------------------------------------------------------
import fingerprint  # Provides build_database() and related helpers


# ---------------------------------------------------------------------------
# Helper: count audio files so we can report stats before building
# ---------------------------------------------------------------------------
def _count_audio_files(directory: str) -> int:
    """
    Walk *directory* and return the number of files whose extension
    (case-insensitive) is a supported audio format (.wav, .mp3, etc.).

    Parameters
    ----------
    directory : str
        Path to the folder to scan.

    Returns
    -------
    int
        Number of audio files found.
    """
    count = 0
    for entry in os.listdir(directory):
        # os.path.splitext returns (root, ext); ext includes the leading dot
        _, ext = os.path.splitext(entry)
        if ext.lower() in fingerprint.SUPPORTED_EXTENSIONS:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Argument parser setup
# ---------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Construct and return the ArgumentParser for this CLI tool.

    Two optional arguments are exposed:
        --songs_dir : directory containing audio files  (default: ./songs)
        --output    : path where the pickle database will be saved
                      (default: ./database.pkl)

    Returns
    -------
    argparse.ArgumentParser
        The fully configured parser, ready for .parse_args().
    """
    parser = argparse.ArgumentParser(
        description=(
            "Index all .wav song files in a directory into an audio-"
            "fingerprint database (saved as a .pkl file)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python build_database.py\n"
            "  python build_database.py --songs_dir ./my_music --output ./my_db.pkl\n"
        ),
    )

    # --songs_dir: where to look for audio files
    parser.add_argument(
        "--songs_dir",
        type=str,
        default="./songs",
        help="Path to the directory containing .wav song files (default: ./songs)",
    )

    # --output: where to save the resulting fingerprint database
    parser.add_argument(
        "--output",
        type=str,
        default="./database.pkl",
        help="Path for the output pickle database file (default: ./database.pkl)",
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Parse CLI arguments, validate inputs, build the fingerprint database,
    and print summary statistics.
    """

    # ---- 1. Parse command-line arguments -----------------------------------
    parser = _build_arg_parser()
    args = parser.parse_args()

    songs_dir: str = args.songs_dir   # e.g. "./songs"
    db_path: str   = args.output      # e.g. "./database.pkl"

    # ---- 2. Validate that the songs directory exists -----------------------
    if not os.path.isdir(songs_dir):
        print(f"[ERROR] Songs directory not found: '{songs_dir}'")
        print("        Please supply a valid path via --songs_dir.")
        sys.exit(1)

    # ---- 3. Count audio files before we start ------------------------------
    num_audio = _count_audio_files(songs_dir)
    if num_audio == 0:
        print(f"[WARNING] No audio files found in '{songs_dir}'. Nothing to index.")
        sys.exit(0)

    # ---- 4. Print a brief header -------------------------------------------
    print("=" * 60)
    print("  Audio Fingerprint Database Builder")
    print("=" * 60)
    print(f"  Songs directory : {os.path.abspath(songs_dir)}")
    print(f"  Output database : {os.path.abspath(db_path)}")
    print(f"  Audio files found: {num_audio}")
    print("=" * 60)
    print()  # blank line for readability

    # ---- 5. Build the database (this is the expensive step) ----------------
    #
    # fingerprint.build_database() will:
    #   • Read each .wav file in songs_dir
    #   • Compute a spectrogram (nperseg=1024, noverlap=512 by default)
    #   • Detect spectral peaks using a local-maximum filter
    #   • Generate combinatorial fingerprint hashes (fan_value=15)
    #   • Store (hash → [(song_id, time_offset), ...]) in a dict
    #   • Pickle the dict to db_path
    #   • Return the dict
    #
    # We time the whole process so we can report wall-clock duration.

    start_time = time.time()

    database = fingerprint.build_database(
        songs_dir=songs_dir,
        db_path=db_path,
    )

    elapsed = time.time() - start_time

    # ---- 6. Compute and display summary statistics -------------------------
    #
    # The database returned by fingerprint.build_database() is a dict with:
    #   "paired_index" : {hash_tuple: [(song_id, offset), ...]}
    #   "single_index" : {hash_tuple: [(song_id, offset), ...]}
    #   "song_names"   : {song_id: "name_without_ext"}
    #   "params"       : {param_name: value}

    paired_index = database.get("paired_index", {})
    single_index = database.get("single_index", {})
    song_names = database.get("song_names", {})

    # Count unique hashes and total entries for the paired index
    num_hashes = len(paired_index)
    num_entries = sum(len(v) for v in paired_index.values())

    # Number of songs indexed
    unique_songs = song_names

    # Database file size on disk (if it was successfully written)
    db_size_str = "N/A"
    if os.path.isfile(db_path):
        db_size_bytes = os.path.getsize(db_path)
        # Format size in a human-friendly way
        if db_size_bytes < 1024:
            db_size_str = f"{db_size_bytes} B"
        elif db_size_bytes < 1024 ** 2:
            db_size_str = f"{db_size_bytes / 1024:.1f} KB"
        else:
            db_size_str = f"{db_size_bytes / (1024 ** 2):.2f} MB"

    # ---- 7. Print the summary banner ---------------------------------------
    print()
    print("=" * 60)
    print("  Database Build Complete!")
    print("=" * 60)
    print(f"  Songs indexed       : {len(unique_songs)}")
    print(f"  Unique hashes       : {num_hashes:,}")
    print(f"  Total entries       : {num_entries:,}")
    print(f"  Database file size  : {db_size_str}")
    print(f"  Time elapsed        : {elapsed:.2f} seconds")
    print(f"  Saved to            : {os.path.abspath(db_path)}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Standard Python idiom: only run main() when this file is executed directly,
# not when it is imported as a module.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
