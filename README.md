# Harmonic Accompaniment Recommendation with Melody Tracking

This project aims to develop a system that generates real-time harmonic accompaniment for a one-line melody.

## Pipeline Overview

### 1. Data Extraction and Cleaning

Run the script `data/*/extract.py` to process and clean the raw data. This will generate `.npz` files for each song from the source dataset.

### 2. Data Augmentation and Preparation

Use the script `data/prepare_training_data.py` to prepare the full training and validation datasets by augmenting the raw data.

### 3. Model Training

Train the model using the prepared dataset.

- To monitor the training progress and visualize the loss, use **TensorBoard**. Start it with the following command:

  ```bash
  python3 -m tensorboard.main --logdir=runs --port=6006
  ```

- Visit [http://localhost:6006](http://localhost:6006) in your browser to view the logs and charts.

### 4. Inference and Evaluation

- **Choose a Song for Inference**  
   Select a song for inference and compare the predicted chords with the ground truth labels.

- **Listen to the MIDI Output**  
   You can listen to the MIDI output by visiting [MIDI Player](https://midiplayer.ehubsoft.net/).

- **Play the MIDI Output Yourself**  
   Alternatively, use `demo.py` to generate and play the MIDI output directly on your machine.
