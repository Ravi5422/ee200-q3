import os
import fingerprint as fp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

songs_dir = "/Users/ravijani/Downloads/ee200_q3/songs"
thumbs_dir = "/Users/ravijani/Downloads/ee200_q3/thumbnails"
os.makedirs(thumbs_dir, exist_ok=True)

plt.style.use('dark_background')

# We only need tiny plots. Disable axes, padding, everything to make it fast.
audio_files = sorted([f for f in os.listdir(songs_dir) if os.path.splitext(f.lower())[1] in fp.SUPPORTED_EXTENSIONS])

for idx, fname in enumerate(audio_files):
    name = os.path.splitext(fname)[0]
    out_path = os.path.join(thumbs_dir, f"{name}.png")
    if os.path.exists(out_path):
        continue
    
    print(f"Generating thumbnail {idx+1}/{len(audio_files)}: {name}")
    try:
        audio, sr = fp.load_audio(os.path.join(songs_dir, fname))
        f, t, Sxx = fp.compute_spectrogram(audio, sr)
        peaks = fp.find_peaks(Sxx)
        
        # Plot just the peaks for the thumbnail
        fig, ax = plt.subplots(figsize=(2, 1), dpi=100)
        ax.set_facecolor('#161b22')
        fig.patch.set_facecolor('#161b22')
        
        peak_times = [t[p[0]] for p in peaks if p[0] < len(t)]
        peak_freqs = [f[p[1]] for p in peaks if p[1] < len(f)]
        
        # Give each song a random but consistent vibrant color based on its name
        # We use a hash to ensure the color stays exactly the same across reruns
        hue = (hash(name) % 1000) / 1000.0
        dot_color = cm.hsv(hue)
        
        ax.scatter(peak_times, peak_freqs, color=dot_color, s=0.5, alpha=0.8)
        
        ax.set_xlim(0, max(t) if len(t) > 0 else 1)
        ax.set_ylim(0, min(8000, f[-1]))
        
        ax.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        plt.savefig(out_path, format='png', bbox_inches='tight', pad_inches=0, facecolor='#161b22')
        plt.close(fig)
    except Exception as e:
        print(f"Error on {fname}: {e}")

print("Done generating thumbnails.")
