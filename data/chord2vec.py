import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from mpl_toolkits.mplot3d import Axes3D
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import networkx as nx
from utils.constants import CHORD_CLASSES_ALL, NUM_QUALITIES_ALL, QUALITIES_ALL, BATCH_SIZE
from torch.utils.data import DataLoader
from AR.ar_transformer import MusicDataset


########################################
# CONFIG
########################################

EMBED_DIM = 32
OUTPUT_DIR = "data"
EMBED_PATH = f"token_embeddings_{EMBED_DIM}d.json"
TRANSITION_PATH = f"transition_graph.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)


########################################
# 1. Extract sequences from loader
########################################

def extract_sequences(train_loader):

    sentences = []

    for _, target in train_loader:

        target = target.cpu().numpy()

        for seq in target:
            if len(seq) > 1:
                sentences.append([int(x) for x in seq])

    return sentences


########################################
# 2. Train Word2Vec
########################################

def train_word2vec(sentences):

    model = Word2Vec(
        sentences=sentences,
        vector_size=EMBED_DIM,
        window=3,
        min_count=1,
        workers=4
    )

    return model


########################################
# 3. Save embeddings
########################################

def save_embeddings(model):

    tokens = list(model.wv.index_to_key)

    data = []

    for token in tokens:

        idx = int(token)

        data.append({
            "idx": idx,
            "label": CHORD_CLASSES_ALL[idx],
            "mod": idx % NUM_QUALITIES_ALL,
            "embedding": model.wv[token].tolist()
        })

    path = os.path.join(OUTPUT_DIR, EMBED_PATH)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    print("Saved embeddings →", path)

    return tokens


########################################
# 4. Build transition counts
########################################

def build_transition_graph(sentences):

    edge_counts = {}

    for seq in sentences:

        for i in range(len(seq) - 1):

            a = seq[i]
            b = seq[i+1]

            edge_counts[(a,b)] = edge_counts.get((a,b),0) + 1

    return edge_counts


########################################
# 5. Save transition graph
########################################

def save_transition_graph(edge_counts):

    edges = []

    for (a,b), w in edge_counts.items():

        a_idx = int(a)
        b_idx = int(b)

        edges.append({
            "source_idx": a_idx,
            "source_label": CHORD_CLASSES_ALL[a_idx],
            "target_idx": b_idx,
            "target_label": CHORD_CLASSES_ALL[b_idx],
            "weight": w,
            "source_mod": a_idx % NUM_QUALITIES_ALL,
            "target_mod": b_idx % NUM_QUALITIES_ALL
        })

    path = os.path.join(OUTPUT_DIR, TRANSITION_PATH)

    with open(path, "w") as f:
        json.dump(edges, f, indent=2)

    print("Saved transitions →", path)

########################################
# MAIN PIPELINE
########################################

def analyze_sequences(train_loader):

    print("Extracting sequences...")
    sentences = extract_sequences(train_loader)

    print("Training Word2Vec...")
    model = train_word2vec(sentences)

    print("Saving embeddings...")
    tokens = save_embeddings(model)

    print("Building transition counts...")
    edge_counts = build_transition_graph(sentences)

    print("Saving transition graph...")
    save_transition_graph(edge_counts)

    print("Done. Results saved to:", OUTPUT_DIR)


########################################
# Load embeddings
########################################

def load_embeddings(path):

    with open(path) as f:
        data = json.load(f)

    tokens = []
    labels = []
    mod = []
    embeddings = []

    for item in data:
        tokens.append(item["idx"])
        labels.append(item["label"])
        mod.append(item["mod"])
        embeddings.append(item["embedding"])

    return (
        tokens,
        labels,
        mod,
        np.array(embeddings)
    )


########################################
# Load transitions
########################################

def load_transitions(path):

    with open(path) as f:
        return json.load(f)


########################################
# 3D PCA
########################################

def compute_pca_3d(embeddings):

    pca = PCA(n_components=6)
    pca.fit(embeddings)

    print("Explained variance ratios:")
    for i, v in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {v:.4f}")

    # still return 3D coords for plotting
    coords = pca.transform(embeddings)[:, :3]
    return coords


########################################
# Plot 3D embeddings
########################################

def plot_embeddings_3d(coords, labels, mod):

    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    coords = np.array(coords)
    mod = np.array(mod)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection="3d")

    unique_mods = np.unique(mod)

    group_artists = {}

    # plot each colour group
    for m in unique_mods:
        idx = np.where(mod == m)[0]
        group_coords = coords[idx]

        scatters, texts, lines = [], [], []

        for i in idx:
            x, y, z = coords[i]

            sc = ax.scatter(x, y, z, color=cmap(m))
            txt = ax.text(x, y, z, labels[i], fontsize=8)

            scatters.append(sc)
            texts.append(txt)

        # nearest neighbour lines
        for i, p in enumerate(group_coords):

            dists = np.linalg.norm(group_coords - p, axis=1)
            dists[i] = np.inf
            nearest = np.argsort(dists)[:2]

            for n in nearest:
                p2 = group_coords[n]

                ln, = ax.plot(
                    [p[0], p2[0]],
                    [p[1], p2[1]],
                    [p[2], p2[2]],
                    color=cmap(m),
                    alpha=0.4
                )
                lines.append(ln)

        group_artists[m] = scatters + texts + lines

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title("Chord Embeddings (3D PCA)")

    # checkbox UI
    rax = plt.axes([0.02, 0.25, 0.18, 0.5])

    checkbox_labels = [QUALITIES_ALL[m] for m in unique_mods]
    visibility = [True] * len(unique_mods)

    check = CheckButtons(rax, checkbox_labels, visibility)

    # colour checkbox labels
    for lbl, m in zip(check.labels, unique_mods):
        lbl.set_color(cmap(m))

    def toggle(label):
        m = unique_mods[checkbox_labels.index(label)]
        artists = group_artists[m]

        visible = not artists[0].get_visible()
        for a in artists:
            a.set_visible(visible)

        plt.draw()

    check.on_clicked(toggle)

    plt.show()

########################################
# Build transition graph
########################################

def plot_transition_graph(transitions, coords, labels, min_weight=20000, top_n=None):

    if top_n:
        transitions = sorted(transitions, key=lambda x: x["weight"], reverse=True)[:top_n]

    G = nx.DiGraph()
    node_mod = {}

    # ---- build graph ----
    for edge in transitions:

        w = edge["weight"]
        if w < min_weight:
            continue

        src = edge["source_label"]
        tgt = edge["target_label"]

        if src == tgt:
            continue

        G.add_edge(src, tgt, weight=w)

        node_mod[src] = edge["source_mod"]
        node_mod[tgt] = edge["target_mod"]

    if len(G.nodes) == 0:
        print("No edges passed filter")
        return

    # only qualities actually present
    unique_mods = sorted(set(node_mod.values()))

    # ---- PCA positions ----
    pos = {labels[i]: (coords[i][0], coords[i][1]) for i in range(len(labels))}

    cmap = plt.cm.get_cmap("tab20", NUM_QUALITIES_ALL)

    fig, ax = plt.subplots(figsize=(9,7))

    node_artists = {}
    edge_artists = []

    # ---- draw nodes ----
    for m in unique_mods:

        nodes = [n for n in G.nodes if node_mod[n] == m]
        if not nodes:
            continue

        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]

        scat = ax.scatter(xs, ys, color=cmap(m), s=200)

        texts = []
        for n in nodes:
            x, y = pos[n]
            texts.append(ax.text(x, y, n, fontsize=6))

        node_artists[m] = [scat] + texts

    # ---- edge widths ----
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_weight = max(weights)

    # ---- draw edges ----
    
    for u, v in G.edges():

        w = G[u][v]["weight"]
        width = 0.5 + 5 * (w / max_weight)

        if G.has_edge(v, u):
            rad = 0.1
        else:
            rad = 0.0

        src_mod = node_mod[u]
        tgt_mod = node_mod[v]

        color = cmap(src_mod)

        edge = nx.draw_networkx_edges(
            G,
            pos,
            edgelist=[(u, v)],
            width=width,
            edge_color=[color],
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
            alpha=0.85,
            ax=ax
        )

        edge_artists.append({
            "artists": edge,
            "src_mod": src_mod,
            "tgt_mod": tgt_mod
        })

    ax.set_title("Chord Transition Graph (PCA Layout)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axis("equal")

    # ---- checkbox UI ----
    rax = plt.axes([0.02, 0.1, 0.1, 0.8])

    checkbox_labels = [QUALITIES_ALL[m] for m in unique_mods]
    visibility = [True] * len(unique_mods)

    check = CheckButtons(rax, checkbox_labels, visibility)

    for lbl, m in zip(check.labels, unique_mods):
        lbl.set_color(cmap(m))

    visible_mods = {m: True for m in unique_mods}

    def update_edges():
        for e in edge_artists:

            visible = visible_mods[e["src_mod"]] and visible_mods[e["tgt_mod"]]

            for a in e["artists"]:
                a.set_visible(visible)

    def toggle(label):

        m = unique_mods[checkbox_labels.index(label)]

        visible_mods[m] = not visible_mods[m]

        for a in node_artists[m]:
            a.set_visible(visible_mods[m])

        update_edges()

        plt.draw()

    check.on_clicked(toggle)

    plt.tight_layout()
    plt.show()

########################################
# MAIN ENTRY
########################################

if __name__ == "__main__":

    # train_data = MusicDataset("data/data_train.npz")
    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    # analyze_sequences(train_loader)

    tokens, labels, mod, embeddings = load_embeddings(os.path.join(OUTPUT_DIR, EMBED_PATH))
    coords = compute_pca_3d(embeddings)
    plot_embeddings_3d(coords, labels, mod)

    transitions = load_transitions(os.path.join(OUTPUT_DIR, TRANSITION_PATH))
    plot_transition_graph(transitions, coords, labels)