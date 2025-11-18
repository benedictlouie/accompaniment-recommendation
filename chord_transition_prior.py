import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from constants import QUALITY_SIMPLIFIER, QUALITIES, FIFTHS, FIFTHS_INDEX, FLAT_TO_SHARP, FIFTHS_CHORD_LIST, NUM_CLASSES
from matplotlib.colors import LogNorm
from collections import Counter

def simplify_chord(chord_str):
    if chord_str == "N" or chord_str is None:
        return "N"
    chord_str = chord_str.split('/')[0]
    root, qual = chord_str.split(':')
    if root.endswith('b'):
        root = FLAT_TO_SHARP.get(root, root)
    qual = QUALITY_SIMPLIFIER.get(qual, qual)
    if qual not in QUALITIES:
        qual = 'maj'
    return f"{root}:{qual}"

def get_bar_chords(strong_beats: np.ndarray, chord_array: np.ndarray):
    """
    Finds the dominant chord for each bar segment using maximum vote, 
    dynamically adjusting to bar lengths defined by the '1's in strong_beats.
    """
    
    # 1. Identify all indices where a new bar starts
    one_indices = np.where(strong_beats == 1)[0]
    
    if len(one_indices) < 1:
        print("Error: 'strong_beats' must contain at least one '1'.")
        return []

    simplified_chords = []
    
    # 2. Iterate through segments defined by consecutive '1' indices
    # The segments are chord_array[start:end]
    for j in range(len(one_indices)):
        start_index = one_indices[j]
    
        if j + 1 < len(one_indices):
            end_index = one_indices[j+1]
        else:
            end_index = len(chord_array) 

        current_bar_segment = chord_array[start_index : end_index]
        
        if len(current_bar_segment) == 0:
            continue
            
        # 3. Maximum Vote Calculation
        counts = Counter(current_bar_segment)
        max_frequency = max(counts.values())

        # 4. Select the first chord in the segment that matches the max frequency (tie-breaker)
        dominant_chord = None
        for chord in current_bar_segment:
            if counts[chord] == max_frequency:
                dominant_chord = chord
                break
        
        if dominant_chord is not None:
            simplified_chords.append(simplify_chord(dominant_chord))
        
    return simplified_chords

if __name__ == "__main__":
    transitions = defaultdict(lambda: defaultdict(int))

    for song_num in range(1, 301):
        song_num_str = f"{song_num:03d}"
        npz_path = f'pop/melody_chords/{song_num_str}.npz'
        data = np.load(npz_path, allow_pickle=True)
        
        strong_beats = data['strong_beats']
        chords = data["chords"]
        chords = get_bar_chords(strong_beats, chords)
        
        # Record transitions
        for i in range(len(chords) - 1):
            current_chord = simplify_chord(chords[i])
            next_chord = simplify_chord(chords[i + 1])

            # for i in range(-6, 6):
            #     chord_from = "N"
            #     chord_to = "N"
            #     if current_chord != "N":
            #         root_from, quality_from = current_chord.split(':')
            #         root_from = (FIFTHS_INDEX[root_from] + i) % 12
            #         chord_from = FIFTHS[(root_from + i) % 12] + ':' + quality_from
            #     if next_chord != "N":
            #         root_to, quality_to = next_chord.split(':')
            #         root_to = (FIFTHS_INDEX[root_to] + i) % 12
            #         chord_to = FIFTHS[(root_to + i) % 12] + ':' + quality_to
            #     transitions[chord_from][chord_to] += 1
            
            transitions[current_chord][next_chord] += 1


    # Step 2: Create transition matrix
    transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))

    for i, chord_from in enumerate(FIFTHS_CHORD_LIST):
        total = sum(transitions[chord_from].values()) + 1
        for j, chord_to in enumerate(FIFTHS_CHORD_LIST):
            transition_matrix[i, j] = transitions[chord_from][chord_to] / total

    np.save('chord_transition_matrix.npy', transition_matrix)

    # Step 3: Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        transition_matrix + 1e-12,
        xticklabels=FIFTHS_CHORD_LIST,
        yticklabels=FIFTHS_CHORD_LIST,
        cmap="Blues",
        norm=LogNorm(vmin=transition_matrix.min() + 1e-6, vmax=transition_matrix.max()),
        annot=False
    )
    plt.xlabel("Next Chord")
    plt.ylabel("Current Chord")
    plt.title("Chord Transition Probability Heatmap")
    plt.show()
