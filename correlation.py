import numpy as np
import matplotlib.pyplot as plt

def average_correlate(melody1, melody2):
    if len(melody1) < len(melody2):
        melody1, melody2 = melody2, melody1
    l1 = len(melody1)
    l2 = len(melody2)
    melody1_centered = melody1 - np.mean(melody1)
    melody2_centered = melody2 - np.mean(melody2)
    autocorr = np.array([])

    for i in range(-l2, l1):
        subarray = melody1_centered[max(0, i): min(l1, l2+i)] * melody2_centered[max(0, -i): min(l2, l1-i)]
        val = np.average(subarray)
        autocorr = np.append(autocorr, val)
    return np.arange(-l2, l1), autocorr

def get_song(song_num):
    song_num_str = f"{song_num:03d}"
    npz_path = f'pop/melody_chords/{song_num_str}.npz'
    data = np.load(npz_path, allow_pickle=True)
    strong_beats = data['strong_beats']
    melody = data["melody"].flatten()
    chords = data["chords"]
    return strong_beats, melody, chords

_, melody1, _ = get_song(330)
_, melody2, _ = get_song(609)
shifts, autocorr = average_correlate(melody1, melody1)

plt.figure(figsize=(8,4))
plt.plot(shifts, autocorr)
plt.xlabel('Time Shift')
plt.ylabel('Autocorrelation')
plt.title('Time-shifted Autocorrelation of Melody')
plt.grid(True)
plt.show()

