import matplotlib.pyplot as plt
from utils.constants import FIFTHS, FIFTHS_CHORD_INDICES

def plot_chords_over_time(*chord_sequences):
    """
    Plot one or more chord sequences on the same graph.
    Each sequence is plotted as a line (with markers).
    """
    plt.figure(figsize=(10, 5))

    for seq_idx, chords in enumerate(chord_sequences):
        times = range(len(chords))
        y_positions = []
        for chord in chords:
            y_pos = FIFTHS_CHORD_INDICES[chord] / 2 - .5
            if y_pos < 0: y_pos = -1
            y_positions.append(y_pos)
        plt.plot(times, y_positions, 'o-', lw=2, label=f"Sequence {seq_idx + 1}")

    # Add -1 ("N") to the y-axis labels
    plt.yticks([-1] + list(range(len(FIFTHS))), ["N"] + FIFTHS)
    plt.xlabel("Time (index)")
    plt.ylabel("Keys (ordered by circle of fifths)")
    plt.title("Chord Progressions vs Time")
    plt.grid(True)
    plt.legend()
    plt.show()
