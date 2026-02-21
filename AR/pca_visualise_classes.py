import torch
import torch.nn as nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from typer import prompt
matplotlib.use("QtAgg")

from matplotlib.widgets import RangeSlider, Button

from sklearn.decomposition import PCA
import os
from AR.ar_transformer import TransformerModel
from utils.constants import INPUT_DIM, NUM_CLASSES_ALL, DEVICE, CHORD_CLASSES_ALL

# ----------------------------
# Load Model
# ----------------------------

def load_model(model_path, input_dim, output_dim):
    model = TransformerModel(input_dim, output_dim).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

# ----------------------------
# PCA + 3D Plot
# ----------------------------

def visualize_class_weights_3d(model):
    # Extract final classification layer weights
    # Shape: [num_classes, d_model]
    weights = model.fc_out.weight.detach().cpu().numpy()

    print("Weight matrix shape:", weights.shape)

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = reduced[:, 0]
    ys = reduced[:, 1]
    zs = reduced[:, 2]

    ax.scatter(xs, ys, zs, c=np.arange(len(xs))%14, cmap=plt.cm.get_cmap("tab20", 12), s=60)

    # Annotate class indices
    for i in range(NUM_CLASSES_ALL):
        ax.text(xs[i], ys[i], zs[i], CHORD_CLASSES_ALL[i], size=8)

    ax.set_title("Chord Similarity")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()

# ----------------------------
# PCA + 2D Plot (PC2 vs PC3, color = PC1)
# ----------------------------

def visualize_class_weights_2d(model):
    weights = model.fc_out.weight.detach().cpu().numpy()

    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)

    pc1 = reduced[:, 0]
    pc2 = reduced[:, 1]
    pc3 = reduced[:, 2]

    # Convert (PC2, PC3) → phase angle
    phase = np.arctan2(pc3, pc2)   # range: [-pi, pi]

    # Optional: radius (strength of harmonic structure)
    radius = np.sqrt(pc2**2 + pc3**2)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(pc1, phase, c=radius, cmap="viridis", s=80)

    cbar = plt.colorbar(scatter)
    cbar.set_label("Radius (sqrt(PC2² + PC3²))")

    for i in range(NUM_CLASSES_ALL):
        plt.text(pc1[i], phase[i], CHORD_CLASSES_ALL[i], fontsize=8)

    plt.title("PC1 vs Phase(PC2, PC3)")
    plt.xlabel("PC1")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_class_weights_4d(model):
    
    # Extract weights
    weights = model.fc_out.weight.detach().cpu().numpy()
    print("Weight matrix shape:", weights.shape)

    # PCA → 6D
    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    pc1 = reduced[:, 0]
    pc2 = reduced[:, 1]
    pc3 = reduced[:, 2]
    pc4 = reduced[:, 3]

    # Cylindrical transform
    r = np.sqrt(pc3**2 + pc4**2)
    theta = np.arctan2(pc4, pc3)   # phase in [-π, π]
    percentiles = np.argsort(np.argsort(r)) / (len(r) - 1)

    # 3D Plot
    fig = plt.figure(figsize=(18, 16), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(pc1, pc2, theta, c=percentiles, cmap="viridis", s=20, alpha=0.9)
    ax.set_box_aspect([1,1,0.8])
    fig.subplots_adjust(left=0.05, right=0.85, top=0.95, bottom=0.05)

    cbar = plt.colorbar(sc, shrink=0.7, pad=0.1)
    cbar.set_label("Percentile of Radius √(PC3² + PC4²)")

    # Labels (optional — remove if cluttered)
    for i in range(NUM_CLASSES_ALL):
        ax.text(pc1[i], pc2[i], theta[i], CHORD_CLASSES_ALL[i], size=4)

    ax.set_title("Chord Embedding", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Phase θ (PC3/PC4)")
    plt.tight_layout()
    plt.show()

def progressive_pca(model, n_steps=5):
    # ----------------------------
    # Extract weights
    # ----------------------------
    weights = model.fc_out.weight.detach().cpu().numpy()

    # Stable color assignment (ONLY ONCE)
    num_points = weights.shape[0]
    base_colors = np.arange(num_points) % 14
    cmap = plt.cm.get_cmap("tab20", 14)

    # Working copies
    current_weights = weights.copy()
    current_labels = np.array(CHORD_CLASSES_ALL)
    current_colors = base_colors.copy()

    for step in range(n_steps):
        if len(current_weights) < 2:
            print("Too few classes remaining. Stopping.")
            break

        print(f"\n--- PCA Step {step + 1} ---")

        # Recompute PCA
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(current_weights)

        x = reduced[:, 0]
        y = reduced[:, 1]

        # ----------------------------
        # Plot
        # ----------------------------
        plt.figure(figsize=(8, 7))
        plt.scatter(
            x,
            y,
            c=current_colors,
            cmap=cmap,
            s=80,
            edgecolor="k",
            linewidth=0.5,
        )

        for i, label in enumerate(current_labels):
            plt.text(x[i], y[i], label, fontsize=8, alpha=0.75)

        plt.title(f"Progressive PCA Step {step + 1}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        # ----------------------------
        # User range input
        # ----------------------------
        def get_float(prompt):
            while True:
                try:
                    return float(input(prompt))
                except ValueError:
                    print("Invalid input. Please enter a valid number.")

        xmin = get_float("Enter xmin: ")
        xmax = get_float("Enter xmax: ")

        mask = (x >= xmin) & (x <= xmax)

        # Filter EVERYTHING consistently
        current_weights = current_weights[mask]
        current_labels = current_labels[mask]
        current_colors = current_colors[mask]

        print(f"Remaining classes: {len(current_labels)}")        

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    MODEL_PATH = "checkpoints/transformer_model.pth"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found!")

    model = load_model(MODEL_PATH, INPUT_DIM, NUM_CLASSES_ALL)
    progressive_pca(model)
