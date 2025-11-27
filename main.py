import os
import numpy as np
from tensorflow import keras
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from utils.constants import OUT_DIR

from functions.data_preparation import (
    load_umist,
)
from functions.data_splitting import (
    stratified_split_and_scale,
    plot_split_distribution,
)
from functions.dimensionality_reduction import (
    run_pca,
    run_autoencoder,
)
from functions.clustering import (
    run_clustering,
    evaluate_clustering,
)
from functions.neural_network import (
    train_classifier,
)


os.makedirs(OUT_DIR, exist_ok=True)


def main():
    # ---------- Step 1: Load UMIST ----------
    print("=== Step 1: Load UMIST dataset ===")
    images, labels, df = load_umist("datasets/umist_cropped.mat")
    N, H, W = images.shape
    X = images.reshape(N, -1)

    # ---------- Step 2: Stratified split + scaling ----------
    print("\n=== Step 2: Stratified split & scaling ===")
    (X_train_sc, X_val_sc, X_test_sc), (y_train, y_val, y_test), scaler = (
        stratified_split_and_scale(X, labels)
    )

    # For visualizing test images later
    # 把 images 拆分为与特征一致的顺序
    X_train, X_test_tmp, img_train, img_test_tmp, y_train_raw, y_test_raw = (
        train_test_split(
            X, images, labels, test_size=0.2, stratify=labels, random_state=42
        )
    )
    rel_val = 0.2 / 0.8
    X_train_tmp, X_val_tmp, img_train_tmp, img_val_tmp, y_train_tmp, y_val_tmp = (
        train_test_split(
            X_train,
            img_train,
            y_train_raw,
            test_size=rel_val,
            stratify=y_train_raw,
            random_state=42,
        )
    )
    # 此处只需要 test 对应图像
    images_test = img_test_tmp

    # plot class distribution
    split_plot_path = os.path.join(OUT_DIR, "split_class_distribution.png")
    plot_split_distribution(y_train, y_val, y_test, split_plot_path)

    # ---------- Step 3: Dimensionality reduction ----------
    print("\n=== Step 3: Dimensionality reduction (PCA + AE) ===")
    (Ztr_pca, Zva_pca, Zte_pca), pca_main = run_pca(
        X_train_sc, X_val_sc, X_test_sc, n_components_list=(10, 20, 50, 100)
    )

    (Ztr_ae, Zva_ae, Zte_ae), encoder, autoencoder = run_autoencoder(
        X_train_sc, X_val_sc, X_test_sc, latent_dim=64, epochs=40, batch_size=64
    )

    # ---------- Step 4: Clustering on PCA features ----------
    print("\n=== Step 4: Clustering on PCA features ===")
    n_classes = len(np.unique(labels))
    kmeans_pca, k_tr, k_te = run_clustering(
        Ztr_pca, Zte_pca, y_train, y_test, n_clusters=n_classes, prefix="pca"
    )

    # 你也可以选择在 AE 特征上再次聚类，这里示范一次 KMeans：
    print("\n=== Optional: Clustering on AE latent features (KMeans) ===")
    kmeans_ae = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    k_tr_ae = kmeans_ae.fit_predict(Ztr_ae)
    evaluate_clustering(Ztr_ae, y_train, k_tr_ae, "AE-KMeans")

    # ---------- Step 5: Supervised NN classifier ----------
    print("\n=== Step 5: Supervised NN classifier ===")

    # Option A: use PCA features only
    model_pca = train_classifier(
        Ztr_pca,
        y_train,
        Zva_pca,
        y_val,
        Zte_pca,
        y_test,
        images_test,
        n_classes=n_classes,
        feature_name="PCA",
    )

    # Option B (optional): concat PCA features + cluster one-hot
    # 这里示例如何将 KMeans 聚类结果当作额外数值特征
    k_tr_onehot = keras.utils.to_categorical(k_tr, num_classes=n_classes)
    k_va = kmeans_pca.predict(Zva_pca)
    k_va_onehot = keras.utils.to_categorical(k_va, num_classes=n_classes)
    k_te_onehot = keras.utils.to_categorical(k_te, num_classes=n_classes)

    Ztr_plus = np.concatenate([Ztr_pca, k_tr_onehot], axis=1)
    Zva_plus = np.concatenate([Zva_pca, k_va_onehot], axis=1)
    Zte_plus = np.concatenate([Zte_pca, k_te_onehot], axis=1)

    model_pca_cluster = train_classifier(
        Ztr_plus,
        y_train,
        Zva_plus,
        y_val,
        Zte_plus,
        y_test,
        images_test,
        n_classes=n_classes,
        feature_name="PCA+Cluster",
    )

    print("\n[Done] All steps (1–5) completed. 所有步骤 1–5 已完成。")
    print("Artifacts saved in:", OUT_DIR)


if __name__ == "__main__":
    main()
