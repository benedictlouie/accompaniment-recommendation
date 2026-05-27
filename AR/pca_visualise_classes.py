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
# PCA + Plots
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

def visualize_class_weights_2d(model):
    weights = model.fc_out.weight.detach().cpu().numpy()

    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)

    pc1 = reduced[:, 0]
    pc2 = reduced[:, 1]
    pc3 = reduced[:, 2]

    phase = np.arctan2(pc3, pc2)
    radius = np.sqrt(pc2**2 + pc3**2)

    qualities = np.arange(len(pc1)) % NUM_QUALITIES_ALL

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_axes([0.22, 0.1, 0.65, 0.8])  # leave left margin for checkbox

    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    group_artists = []

    for q in range(NUM_QUALITIES_ALL):
        idxs = np.where(qualities == q)[0]

        xg = pc1[idxs]
        yg = phase[idxs]
        rg = radius[idxs]

        # scatter (color = radius)
        scatter = ax.scatter(xg, yg, c=rg, cmap="viridis", s=20)

        # labels
        texts = []
        for i in idxs:
            t = ax.text(pc1[i], phase[i], CHORD_CLASSES_ALL[i], fontsize=8)
            texts.append(t)

        # nearest neighbour lines
        pts = np.stack([xg, yg], axis=1)
        dists = pairwise_distances(pts)

        lines = []
        for i in range(len(idxs)):
            order = np.argsort(dists[i])[1:3]

            for j in order:
                line, = ax.plot(
                    [xg[i], xg[j]],
                    [yg[i], yg[j]],
                    color=cmap(q),
                    alpha=0.5,
                    linewidth=2.5
                )
                lines.append(line)

        group_artists.append((scatter, texts, lines))

    # colorbar (global)
    cbar = plt.colorbar(group_artists[0][0], ax=ax)
    cbar.set_label("Radius (sqrt(PC2² + PC3²))")

    ax.set_title("PC1 vs Phase(PC2, PC3)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("Phase (radians)")
    ax.grid(True)

    # checkbox UI
    rax = plt.axes([0.02, 0.4, 0.15, 0.4])

    labels = [QUALITIES_ALL[i] for i in range(NUM_QUALITIES_ALL)]
    visibility = [True] * NUM_QUALITIES_ALL

    check = CheckButtons(rax, labels, visibility)

    for i, text in enumerate(check.labels):
        text.set_color(cmap(i))

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

def visualize_class_weights_4d(model):
    weights = model.fc_out.weight.detach().cpu().numpy()
    print("Weight matrix shape:", weights.shape)

    pca = PCA(n_components=6)
    reduced = pca.fit_transform(weights)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    pc1 = reduced[:, 0]
    pc2 = reduced[:, 1]
    pc3 = reduced[:, 2]
    pc4 = reduced[:, 3]

    r = np.sqrt(pc3**2 + pc4**2)
    theta = np.arctan2(pc4, pc3)
    percentiles = np.argsort(np.argsort(r)) / (len(r) - 1)

    qualities = np.arange(len(pc1)) % NUM_QUALITIES_ALL

    fig = plt.figure(figsize=(18, 16), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    group_artists = []

    for q in range(NUM_QUALITIES_ALL):
        idxs = np.where(qualities == q)[0]

        xg = pc1[idxs]
        yg = pc2[idxs]
        zg = theta[idxs]
        rg = percentiles[idxs]

        scatter = ax.scatter(xg, yg, zg, c=rg, cmap="viridis", s=20, alpha=0.9)

        # labels
        texts = []
        for i in idxs:
            t = ax.text(pc1[i], pc2[i], theta[i], CHORD_CLASSES_ALL[i], size=4)
            texts.append(t)

        # nearest neighbour lines
        pts = np.stack([xg, yg, zg], axis=1)
        dists = pairwise_distances(pts)

        lines = []
        for i in range(len(idxs)):
            order = np.argsort(dists[i])[1:3]

            for j in order:
                line, = ax.plot(
                    [xg[i], xg[j]],
                    [yg[i], yg[j]],
                    [zg[i], zg[j]],
                    color=cmap(q),
                    alpha=0.5,
                    linewidth=1
                )
                lines.append(line)

        group_artists.append((scatter, texts, lines))

    # colorbar
    cbar = plt.colorbar(group_artists[0][0], ax=ax, shrink=0.7, pad=0.1)
    cbar.set_label("Percentile of Radius √(PC3² + PC4²)")

    ax.set_title("Chord Embedding", fontsize=14)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("Phase θ (PC3/PC4)")

    # checkbox UI
    rax = plt.axes([0.02, 0.4, 0.15, 0.4])

    labels = [QUALITIES_ALL[i] for i in range(NUM_QUALITIES_ALL)]
    visibility = [True] * NUM_QUALITIES_ALL

    check = CheckButtons(rax, labels, visibility)

    for i, text in enumerate(check.labels):
        text.set_color(cmap(i))

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

        alphas = np.ones(len(x))

        scatter = ax.scatter(
            x,
            y,
            c=current_colors,
            cmap=cmap,
            s=80,
            edgecolor="k",
            linewidth=0.5,
            alpha=None
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

            visible = alphas[mask][0] > 0

            new_alpha = 0 if visible else 1
            alphas[mask] = new_alpha

            scatter.set_alpha(alphas)

            # toggle labels
            for i in range(len(texts)):
                if mask[i]:
                    texts[i].set_visible(not visible)

            # toggle lines
            for line, colour in lines:
                if colour == idx:
                    line.set_visible(not visible)

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
    # progressive_pca(model)
    visualize_class_weights_2d(model)
