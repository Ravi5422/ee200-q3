import os
import pickle
import fingerprint as fp

songs_dir = "/Users/ravijani/Downloads/ee200_q3/songs"
out_path = "/Users/ravijani/Downloads/ee200_q3/constellations.pkl"

constellations = {}

if os.path.exists(songs_dir):
    audio_files = [f for f in os.listdir(songs_dir) if f.endswith('.mp3')]
    for i, f_name in enumerate(audio_files):
        print(f"Extracting {i+1}/{len(audio_files)}: {f_name}")
        song_name = os.path.splitext(f_name)[0]
        audio, sr = fp.load_audio(os.path.join(songs_dir, f_name), sr=22050)
        f, t, Sxx = fp.compute_spectrogram(audio, sr)
        peaks = fp.find_peaks(Sxx)
        # Store just the (frame, bin) coordinates
        constellations[song_name] = [(p[0], p[1]) for p in peaks]

with open(out_path, "wb") as f:
    pickle.dump(constellations, f)

print("Saved constellations.pkl")
