# modified by Al Helal Shourav
# Step 4: Clustering
from sklearn.manifold import TSNE
from collections import Counter  # used for purity calculation
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils.constants import OUT_DIR
import numpy as np
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)


def run_clustering(Z_train, Z_test, y_train, y_test, n_clusters, prefix="pca"):
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    # fit Kmeans on training data
    k_tr = kmeans.fit_predict(Z_train)
    k_te = kmeans.predict(Z_test)

    # evaluate cluster quality

    evaluate_clustering(Z_train, y_train, k_tr, f"{prefix}-KMeans")
    # plot 2d visualization for training and test
    plot_clustering_2d(Z_train, k_tr, y_train, f"{prefix}_kmeans_train.png")
    plot_clustering_2d(Z_test, k_te, y_test, f"{prefix}_kmeans_test.png")
    # print purity analysis for training and test clusters
    cluster_purity(k_tr, y_train, "KMeans (train)")
    cluster_purity(k_te, y_test, "KMeans (test)")

    # Agglomerative (Hierarchical) Clustering
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    # fit agglomerative on train and test set separately
    a_tr = agg.fit_predict(Z_train)
    a_te = agg.fit_predict(
        Z_test
    )  # Note: Agglomerative doesn't have .predict, so fit separately
    evaluate_clustering(Z_train, y_train, a_tr, f"{prefix}-Agglomerative")
    plot_clustering_2d(Z_train, a_tr, y_train, f"{prefix}_agglo_train.png")
    plot_clustering_2d(Z_test, a_te, y_test, f"{prefix}_agglo_test.png")
    cluster_purity(a_tr, y_train, "Agglomerative (train)")
    cluster_purity(a_te, y_test, "Agglomerative (test)")

    return kmeans, k_tr, k_te


def plot_clustering_2d(Z, cluster_labels, true_labels, out_path):
    # Reduce to 2D for visualization
    if Z.shape[1] > 2:
        reducer = PCA(n_components=2)
        Z_2d = reducer.fit_transform(Z)
    else:
        Z_2d = Z
    import os

    plt.figure(figsize=(7, 5))
    # scatter plot each point colored by cluster label
    scatter = plt.scatter(
        Z_2d[:, 0], Z_2d[:, 1], c=cluster_labels, cmap="tab20", alpha=0.7, s=30
    )
    # Set title based on the filename
    plot_label = (
        os.path.splitext(os.path.basename(out_path))[0].replace("_", " ").upper()
    )
    plt.title(f"Clustering Result: {plot_label}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()

    # save the plot to file
    plt.savefig(os.path.join(OUT_DIR, out_path), dpi=150)
    plt.show()  # shows the plot
    plt.close()
    print(f"[SAVE] 2D cluster plot -> {os.path.join(OUT_DIR, out_path)}")


def evaluate_clustering(Z, y_true, cluster_labels, method_name):
    # print clustering evaluation metrics
    print(f"\n=== Clustering Evaluation: {method_name} ===")
    print("Silhouette Score:", silhouette_score(Z, cluster_labels))
    print("Adjusted Rand Index:", adjusted_rand_score(y_true, cluster_labels))
    print(
        "Normalized Mutual Info:", normalized_mutual_info_score(y_true, cluster_labels)
    )


def cluster_purity(cluster_labels, true_labels, method_name):
    n_clusters = len(np.unique(cluster_labels))
    total = len(cluster_labels)
    print(f"\n[Purity Analysis] {method_name}")
    for c in range(n_clusters):
        idx = np.where(cluster_labels == c)[0]
        if len(idx) == 0:
            continue
        true_counts = Counter(true_labels[idx])  # count true labels in this cluster
        most_common = true_counts.most_common(1)[0]
        purity = most_common[1] / len(idx)  # # Purity = fraction of majority label
        print(
            f"Cluster {c}: size={len(idx)}, purity={purity:.2f}, label_counts={dict(true_counts)}"
        )

    # Calculate Overall purity

    majority_sum = sum(
        Counter(
            [
                Counter(true_labels[np.where(cluster_labels == c)[0]]).most_common(1)[
                    0
                ][1]
                for c in range(n_clusters)
            ]
        )
    )
    overall_purity = majority_sum / total
    print(f"Overall purity: {overall_purity:.3f}")
