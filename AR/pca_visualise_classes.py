import torch
import torch.nn as nn
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from typer import prompt
matplotlib.use("QtAgg")

from matplotlib.widgets import RangeSlider, Button

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from matplotlib.widgets import CheckButtons
import os
from AR.ar_transformer import TransformerModel
from utils.constants import INPUT_DIM, QUALITIES_ALL, NUM_CLASSES_ALL, NUM_QUALITIES_ALL, DEVICE, CHORD_CLASSES_ALL

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

    weights = model.fc_out.weight.detach().cpu().numpy()
    print("Weight matrix shape:", weights.shape)

    # PCA
    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    xs = reduced[:, 0]
    ys = reduced[:, 1]
    zs = reduced[:, 2]

    qualities = np.arange(len(xs)) % NUM_QUALITIES_ALL

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    group_artists = []

    for q in range(NUM_QUALITIES_ALL):

        idxs = np.where(qualities == q)[0]

        xg = xs[idxs]
        yg = ys[idxs]
        zg = zs[idxs]

        color = cmap(q)

        scatter = ax.scatter(xg, yg, zg, color=color, s=60)

        # labels
        texts = []
        for i in idxs:
            t = ax.text(xs[i], ys[i], zs[i], CHORD_CLASSES_ALL[i], size=8)
            texts.append(t)

        # nearest neighbour lines (within group)
        pts = np.stack([xg, yg, zg], axis=1)
        dists = pairwise_distances(pts)

        lines = []

        for i in range(len(idxs)):

            order = np.argsort(dists[i])[1:3]  # 2 closest (skip self)

            for j in order:
                xline = [xg[i], xg[j]]
                yline = [yg[i], yg[j]]
                zline = [zg[i], zg[j]]

                line, = ax.plot(xline, yline, zline, color=color, alpha=0.5, linewidth=1)
                lines.append(line)

        group_artists.append((scatter, texts, lines))

    # axes labels
    ax.set_title("Chord Similarity")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    # checkbox UI
    rax = plt.axes([0.02, 0.4, 0.15, 0.4])

    labels = [QUALITIES_ALL[i] for i in range(NUM_QUALITIES_ALL)]
    visibility = [True] * NUM_QUALITIES_ALL

    check = CheckButtons(rax, labels, visibility)

    # colour checkbox labels and boxes
    for i, text in enumerate(check.labels):
        text.set_color(cmap(i % NUM_QUALITIES_ALL))

    def toggle(label):
        idx = labels.index(label)
        scatter, texts, lines = group_artists[idx]

        visible = not scatter.get_visible()

        scatter.set_visible(visible)

        for t in texts:
            t.set_visible(visible)

        for l in lines:
            l.set_visible(visible)

        plt.draw()

    check.on_clicked(toggle)

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

    num_points = weights.shape[0]

    # Stable colour assignment
    base_colors = np.arange(num_points) % NUM_QUALITIES_ALL
    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    # Working copies
    current_weights = weights.copy()
    current_labels = np.array(CHORD_CLASSES_ALL)
    current_colors = base_colors.copy()

    for step in range(n_steps):

        if len(current_weights) < 2:
            print("Too few classes remaining. Stopping.")
            break

        print(f"\n--- PCA Step {step + 1} ---")

        # ----------------------------
        # PCA
        # ----------------------------
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(current_weights)

        x = reduced[:, 0]
        y = reduced[:, 1]

        # ----------------------------
        # Figure + checkbox layout
        # ----------------------------
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_axes([0.25, 0.1, 0.7, 0.8])
        rax = fig.add_axes([0.02, 0.2, 0.18, 0.6])

        scatter = ax.scatter(
            x,
            y,
            c=current_colors,
            cmap=cmap,
            s=80,
            edgecolor="k",
            linewidth=0.5,
        )

        # Labels
        texts = []
        for i, label in enumerate(current_labels):
            t = ax.text(x[i], y[i], label, fontsize=8, alpha=0.75)
            texts.append(t)

        ax.set_title(f"Progressive PCA Step {step + 1}")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_aspect("equal", adjustable="box")

        # ----------------------------
        # Draw nearest neighbour lines (same colour)
        # ----------------------------
        coords = np.stack([x, y], axis=1)
        lines = []

        for colour in np.unique(current_colors):

            idx = np.where(current_colors == colour)[0]

            if len(idx) < 2:
                continue

            pts = coords[idx]

            dists = pairwise_distances(pts)
            np.fill_diagonal(dists, np.inf)

            nearest = np.argsort(dists, axis=1)[:, :2]

            for i, neighbours in enumerate(nearest):
                for n in neighbours:
                    p1 = pts[i]
                    p2 = pts[n]

                    line = ax.plot(
                        [p1[0], p2[0]],
                        [p1[1], p2[1]],
                        color=scatter.cmap(scatter.norm(colour)),
                        alpha=0.3,
                        linewidth=1,
                    )[0]

                    lines.append((line, colour))

        # ----------------------------
        # Checkboxes
        # ----------------------------
        labels = QUALITIES_ALL
        visibility = [True] * len(labels)

        check = CheckButtons(rax, labels, visibility)
        for i, text in enumerate(check.labels):
            text.set_color(cmap(i))


        def toggle(label):

            idx = labels.index(label)

            mask = current_colors == idx

            offsets = scatter.get_offsets()

            for i in range(len(offsets)):
                if current_colors[i] == idx:
                    visible = not texts[i].get_visible()
                    texts[i].set_visible(visible)

            # toggle lines
            for line, colour in lines:
                if colour == idx:
                    line.set_visible(not line.get_visible())

            fig.canvas.draw_idle()

        check.on_clicked(toggle)

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
