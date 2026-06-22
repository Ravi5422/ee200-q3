import os
import random
import numpy as np
import soundfile as sf
import librosa

songs_dir = "/Users/ravijani/Downloads/ee200_q3/songs"
queries_dir = "/Users/ravijani/Downloads/ee200_q3/queries"

songs = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
random.seed(42)  # Just so we get consistent random songs
random.shuffle(songs)

# Generate sample6 to sample10
for i in range(6, 11):
    song = songs[i] # Pick different songs
    print(f"Generating sample{i}.wav from {song}...")
    
    # Load 15 seconds starting at 40s
    audio, sr = librosa.load(os.path.join(songs_dir, song), sr=22050, offset=40.0, duration=10.0, mono=True)
    
    # Save as wav
    out_path = os.path.join(queries_dir, f"sample{i}.wav")
    sf.write(out_path, audio, sr)
    print(f"Saved {out_path}")
