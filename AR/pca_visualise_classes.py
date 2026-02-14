import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
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
    pca = PCA(n_components=3)
    reduced = pca.fit_transform(weights)

    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # Plot 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs = reduced[:, 0]
    ys = reduced[:, 1]
    zs = reduced[:, 2]

    ax.scatter(xs, ys, zs, c=np.arange(len(xs)), cmap="tab20", s=60)

    # Annotate class indices
    for i in range(len(xs)):
        if i < len(CHORD_CLASSES_ALL):
            ax.text(xs[i], ys[i], zs[i], CHORD_CLASSES_ALL[i], size=8)
        else:
            ax.text(xs[i], ys[i], zs[i], f"Class {i}", size=8)

    ax.set_title("Chord Similarity")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    plt.show()

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    MODEL_PATH = "checkpoints/transformer_model.pth"
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"{MODEL_PATH} not found!")

    model = load_model(MODEL_PATH, INPUT_DIM, NUM_CLASSES_ALL)
    visualize_class_weights_3d(model)
