import matplotlib.pyplot as plt

# Circle of fifths order (12 positions)
fifths = ["C", "G", "D", "A", "E", "B", "F#", "Db", "Ab", "Eb", "Bb", "F"]
note_to_index = {note: i for i, note in enumerate(fifths)}

def chord_to_position(chord):
    """Return y position index based on circle of fifths (+3 mod for relative minors)."""
    # Handle 'N' (no chord)
    if chord == 'N':
        return -1  # put N below all other chords

    # Expected format: Root:Quality (e.g. "C:maj" or "A:min")
    try:
        root, quality = chord.split(':')
    except ValueError:
        print(f"Warning: Invalid chord format '{chord}', expected 'Root:Quality'")
        return None

    # Normalize enharmonics
    root = {
        'C#': 'Db',
        'D#': 'Eb',
        'G#': 'Ab',
        'A#': 'Bb',
        'Gb': 'F#'
    }.get(root, root)

    if root not in note_to_index:
        print(f"Warning: {root} not in circle of fifths")
        return None

    idx = note_to_index[root]

    # Minor chords belong to the relative major (-3 mod 12)
    if quality == "min":
        idx = (idx - 3) % 12 + 0.3  # offset for visibility

    return idx


def plot_chords_over_time(*chord_sequences):
    """
    Plot one or more chord sequences on the same graph.
    Each sequence is plotted as a line (with markers).
    """
    plt.figure(figsize=(10, 5))

    for seq_idx, chords in enumerate(chord_sequences):
        times = range(len(chords))
        y_positions = [chord_to_position(c) for c in chords]

        plt.plot(times, y_positions, 'o-', lw=2, label=f"Sequence {seq_idx + 1}")

    # Add -1 ("N") to the y-axis labels
    plt.yticks([-1] + list(range(len(fifths))), ["N"] + fifths)
    plt.xlabel("Time (index)")
    plt.ylabel("Keys (ordered by circle of fifths)")
    plt.title("Chord Progressions vs Time (Circle of Fifths Order, with N)")
    plt.grid(True)
    plt.legend()
    plt.show()
