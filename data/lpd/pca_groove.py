import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mplcursors

from data.lpd import storage as lpd_storage


SAMPLE_SIZE = 200000
SAVE_PATH = "report/figures/groove_pca.png"


if __name__ == "__main__":

    # --- load NN model ---
    nn, loops = lpd_storage.load("drums")

    # sklearn stores the fitted dataset here
    X = nn._fit_X

    print("Feature matrix:", X.shape)

    # --- random sampling ---
    if len(X) > SAMPLE_SIZE:
        idx = np.random.choice(len(X), SAMPLE_SIZE, replace=False)
        X = X[idx]

    print("Sampled:", X.shape)

    density = X.mean(axis=1)

    # --- PCA ---
    pca = PCA(n_components=6)
    X_pca = pca.fit_transform(X)

    print("\nFirst 6 explained variances:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {v:.4f}  ({v*100:.1f}%)")

    print("\nPC2 loadings (per onset position):")
    for i, w in enumerate(pca.components_[1]):
        print(f"  step {i:2d}: {w:+.4f}")

    # --- 2D Plot ---
    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=density,
        cmap="viridis",
        s=4,
        alpha=0.4
    )
    plt.colorbar(scatter, ax=ax, label="Onset density (fraction of steps active)")

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
    ax.set_title(f"PCA of Groove Feature Space ({SAMPLE_SIZE:,} bars from LPD-5)\n"
                 r"Feature $\mathbf{f} = [o_1, \ldots, o_{16}] \in \mathbb{R}^{16}$")

    # --- click annotation ---
    cursor = mplcursors.cursor(scatter, hover=False)

    @cursor.connect("add")
    def on_add(sel):
        i = sel.index
        vec = X[i]
        active = np.where(vec > 0)[0].tolist()
        sel.annotation.set_text(f"{active}  d={density[i]:.3f}")

    plt.tight_layout()
    plt.savefig(SAVE_PATH, dpi=150)
    print(f"\nSaved to {SAVE_PATH}")
    plt.show()
