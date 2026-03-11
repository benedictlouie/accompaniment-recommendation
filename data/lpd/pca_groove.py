import joblib
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors


JOBLIB_PATH = "data/lpd/drum_nn.joblib"
SAMPLE_SIZE = 200000


if __name__ == "__main__":

    # --- load NN model ---
    nn, loops = joblib.load(JOBLIB_PATH)

    # sklearn stores the fitted dataset here
    X = nn._fit_X

    print("Original feature matrix:", X.shape)

    # --- random sampling ---
    if len(X) > SAMPLE_SIZE:
        idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        X = X[idx]

    print("Sampled matrix:", X.shape)

    # --- PCA ---
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X)

    print("\nFirst 6 explained variances:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {v:.6f}")

    # --- 2D Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        s=4,
        alpha=0.4
    )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("2D PCA of Groove Feature Space")

    # --- click annotation ---
    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        vec = X[i]

        # first 16 binary features
        binary_indices = np.where(vec[:16] == 1)[0].tolist()

        # 17th feature
        d16 = vec[16]

        text = f"{binary_indices} {d16:.3f}"

        sel.annotation.set_text(text)

    plt.tight_layout()
    plt.show()