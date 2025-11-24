import pygame
import numpy as np
import time

# Initialize pygame
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Settings
bpm = 80
beat_duration = 60 / bpm
beats_per_bar = 4
bar_duration = beat_duration * beats_per_bar

# Load metronome click
click_sound = pygame.mixer.Sound("click.wav")

# Key mapping (1.5 octaves starting from C4)
key_map = {
    'a': 'C4', 'w': 'C#4', 's': 'D4', 'e': 'D#4', 'd': 'E4',
    'f': 'F4', 't': 'F#4', 'g': 'G4', 'y': 'G#4', 'h': 'A4',
    'u': 'A#4', 'j': 'B4', 'k': 'C5', 'o': 'C#5', 'l': 'D5',
    'p': 'D#5', ';': 'E5', "'": 'F5'
}

# Frequencies for each note
note_freqs = {
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63,
    'F4': 349.23, 'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00,
    'A#4': 466.16, 'B4': 493.88, 'C5': 523.25, 'C#5': 554.37, 'D5': 587.33,
    'D#5': 622.25, 'E5': 659.25, 'F5': 698.46
}

# Pre-generate pygame Sounds for all notes (1s long)
def generate_note_sound(freq, duration=1.0):
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = 0.3 * np.sin(2 * np.pi * freq * t)
    wave = np.int16(wave * 32767)
    return pygame.sndarray.make_sound(np.column_stack([wave, wave]))

note_sounds = {note: generate_note_sound(freq) for note, freq in note_freqs.items()}

# Initialize mixer
pygame.mixer.init(frequency=44100, size=-16, channels=1)

# Ensure enough channels for all notes
pygame.mixer.set_num_channels(len(note_freqs))

# Channels for each note
note_channels = {note: pygame.mixer.Channel(i) for i, note in enumerate(note_freqs)}

# Piano window
width, height = 800, 200
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Python Piano Keyboard with Metronome")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50, 50, 50)
BLUE = (0, 0, 255)

# Key dimensions
white_keys = ['C', 'D', 'E', 'F', 'G', 'A', 'B'] * 2
white_key_width = width // len(white_keys)
white_key_height = height
black_key_width = white_key_width // 2
black_key_height = height * 0.6
black_keys = ['C#', 'D#', '', 'F#', 'G#', 'A#', '', 'C#', 'D#', '', 'F#', 'G#', 'A#', '']

# Track notes and times
notes_pressed = {}  # note -> press_time
notes_played = []   # list of (note, start_time, end_time)

# Metronome timer
last_beat_time = time.time()
current_beat = 0
bar_start_time = last_beat_time

# Main loop
running = True
while running:
    current_time = time.time()
    
    # Metronome tick
    if current_time - last_beat_time >= beat_duration:
        click_sound.play()
        last_beat_time = current_time
        current_beat += 1
        
        if current_beat >= beats_per_bar:
            # End of bar: compute proportions
            bar_end_time = current_time
            bar_proportions = {}
            for note, start, end in notes_played:
                # Clip to bar duration
                start = max(start, bar_start_time)
                end = min(end, bar_end_time)
                duration = max(0, end - start)
                if note in bar_proportions:
                    bar_proportions[note] += duration
                else:
                    bar_proportions[note] = duration
            for note in bar_proportions:
                bar_proportions[note] /= bar_duration
            print("Bar completed! Note proportions:", bar_proportions)
            notes_played.clear()
            current_beat = 0
            bar_start_time = current_time
    
    # Draw keys
    screen.fill(GRAY)
    
    # White keys
    for i, key in enumerate(white_keys):
        note_name = key + '4' if i < 7 else key + '5'
        color = BLUE if note_name in notes_pressed else WHITE
        pygame.draw.rect(screen, color, (i * white_key_width, 0, white_key_width, white_key_height))
        pygame.draw.rect(screen, BLACK, (i * white_key_width, 0, white_key_width, white_key_height), 2)
    
    # Black keys
    for i, key in enumerate(black_keys):
        if key != '':
            note_name = key + '4' if i < 7 else key + '5'
            color = BLUE if note_name in notes_pressed else BLACK
            pygame.draw.rect(screen, color, (i * white_key_width + 0.7 * white_key_width, 0, black_key_width, black_key_height))
    
    pygame.display.flip()
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            if key_name in key_map and key_map[key_name] not in notes_pressed:
                note = key_map[key_name]
                notes_pressed[note] = current_time
                note_channels[note].play(note_sounds[note], loops=-1)
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            if key_name in key_map:
                note = key_map[key_name]
                if note in notes_pressed:
                    start_time = notes_pressed.pop(note)
                    notes_played.append((note, start_time, current_time))
                    note_channels[note].stop()
    
    pygame.time.delay(10)

pygame.quit()
