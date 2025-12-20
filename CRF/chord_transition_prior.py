import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from utils.constants import QUALITY_SIMPLIFIER, QUALITIES, FIFTHS, FIFTHS_INDEX, FLAT_TO_SHARP, FIFTHS_CHORD_LIST, NUM_CLASSES, CHORD_CLASSES
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
    transitions = {chord_from: {chord_to: {key: 0 for key in CHORD_CLASSES} for chord_to in CHORD_CLASSES} for chord_from in CHORD_CLASSES}

    for song_num in range(1, 801):
        song_num_str = f"{song_num:03d}"
        npz_path = f'data/pop/melody_chords/{song_num_str}.npz'
        data = np.load(npz_path, allow_pickle=True)
        
        strong_beats = data['strong_beats']
        chords = data["chords"]
        chords = get_bar_chords(strong_beats, chords)

        with open(f'data/pop/POP909/{song_num_str}/key_audio.txt', 'r') as file:
            key = file.readline().strip().split('\t')[2]
            key = simplify_chord(key)

        # Record transitions
        for i in range(len(chords) - 1):
            current_chord = simplify_chord(chords[i])
            next_chord = simplify_chord(chords[i + 1])

            for transpose in range(-6, 6):
                root_key, quality_key = key.split(':')
                root_key = (FIFTHS_INDEX[root_key] + transpose) % 12
                transposed_key = FIFTHS[root_key] + ':' + quality_key

                chord_from = "N"
                chord_to = "N"
                if current_chord != "N":
                    root_from, quality_from = current_chord.split(':')
                    root_from = (FIFTHS_INDEX[root_from] + transpose) % 12
                    chord_from = FIFTHS[root_from] + ':' + quality_from
                if next_chord != "N":
                    root_to, quality_to = next_chord.split(':')
                    root_to = (FIFTHS_INDEX[root_to] + transpose) % 12
                    chord_to = FIFTHS[root_to] + ':' + quality_to
                transitions[chord_from][chord_to][transposed_key] += 1

    # Step 2: Create transition matrix
    transition_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES, NUM_CLASSES-1))

    for i, chord_from in enumerate(FIFTHS_CHORD_LIST):
        for j, chord_to in enumerate(FIFTHS_CHORD_LIST):
            for k, key in enumerate(FIFTHS_CHORD_LIST[:-1]):
                transition_matrix[i, j, k] = transitions[chord_from][chord_to][key]

    np.save('crf/chord_transition_matrix.npy', transition_matrix)

    transition_matrix = np.sum(transition_matrix, axis=2)
    transition_matrix /= np.sum(transition_matrix, axis=1)

    # Step 3: Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        transition_matrix,
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
