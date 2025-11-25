# ==========================================================
# COMP257 Group Project - UMIST Faces Full Pipeline (v2)
# Author: Tian Li
# ==========================================================
# Steps 步骤:
# 1) Load UMIST dataset from .mat and inspect structure
#    从 .mat 文件加载 UMIST 数据集并解析结构
# 2) Stratified train/val/test split + standardization
#    分层采样拆分 train/val/test 并做标准化
# 3) Dimensionality reduction: PCA + Autoencoder
#    降维：PCA + 自编码器
# 4) Clustering on reduced features: K-Means + Agglomerative
#    在降维特征上做聚类（K-Means + 层次聚类）
# 5) Supervised NN classifier (Keras) for face ID
#    用 Keras 建立监督神经网络分类器做人脸识别
# ==========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.io import loadmat

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    classification_report,
    confusion_matrix,
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

OUT_DIR = "outputs_UMIST_v2"
os.makedirs(OUT_DIR, exist_ok=True)


# ----------------------------------------------------------
# 1. Load UMIST .mat helper
# ----------------------------------------------------------
def load_umist(mat_path="umist_cropped.mat"):
    """
    Load UMIST faces from .mat file.
    从 .mat 文件中加载 UMIST 人脸数据。

    Expected structure (typical):
      mat['facedat']  -> 1 x N_person cell array
      each cell: images of one person, shape either:
        (H, W, n_i)  or  (H*W, n_i)

    Returns:
      images: (N, H, W)  raw grayscale images
      labels: (N,)       person id [0..N_person-1]
      df:      Pandas DataFrame with flattened features
    """
    mat = loadmat(mat_path)
    if "facedat" not in mat:
        raise RuntimeError(f"'facedat' not found in {mat_path}. Keys: {mat.keys()}")

    facedat = mat["facedat"]  # usually shape (1, 20)
    n_persons = facedat.shape[1]

    images_list = []
    labels_list = []

    # First pass: infer image height & width from first subject
    first_cell = facedat[0, 0]
    if first_cell.ndim == 3:
        H, W, _ = first_cell.shape
    elif first_cell.ndim == 2:
        # assume (H*W, n_i)
        pixels = first_cell.shape[0]
        # UMIST 通常是 112x92，如果像素数匹配就用；否则粗暴猜一个近似矩形
        if pixels == 112 * 92:
            H, W = 112, 92
        else:
            H = int(np.sqrt(pixels))
            W = pixels // H
    else:
        raise RuntimeError("Unexpected facedat cell dimension")

    print(f"[INFO] Inferred image size: H={H}, W={W}")

    # Iterate over persons
    for pid in range(n_persons):
        cell = facedat[0, pid]
        if cell.ndim == 3:
            # shape (H, W, n_i)
            for j in range(cell.shape[2]):
                img = cell[:, :, j]
                images_list.append(img.astype(np.float32))
                labels_list.append(pid)
        elif cell.ndim == 2:
            # shape (H*W, n_i) or (n_i, H*W)
            if cell.shape[0] == H * W:
                for j in range(cell.shape[1]):
                    img = cell[:, j].reshape(H, W)
                    images_list.append(img.astype(np.float32))
                    labels_list.append(pid)
            elif cell.shape[1] == H * W:
                for j in range(cell.shape[0]):
                    img = cell[j, :].reshape(H, W)
                    images_list.append(img.astype(np.float32))
                    labels_list.append(pid)
            else:
                raise RuntimeError(
                    f"Cannot reshape cell for person {pid}, shape={cell.shape}"
                )
        else:
            raise RuntimeError(f"Unexpected cell ndim={cell.ndim} for person {pid}")

    images = np.stack(images_list, axis=0)  # (N, H, W)
    labels = np.array(labels_list, dtype=np.int32)
    N = images.shape[0]

    # Flatten for ML input (each row is one image)
    X = images.reshape(N, -1)

    # Build DataFrame
    df = pd.DataFrame(X)
    df["person_id"] = labels

    print(f"[INFO] Loaded {N} images from {n_persons} persons.")
    return images, labels, df


# ----------------------------------------------------------
# 2. Stratified train/val/test split + scaling
# ----------------------------------------------------------
def stratified_split_and_scale(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    X: (N, D) flattened images
    y: (N,) labels
    使用分层采样做 train/val/test 拆分，并对特征做标准化。
    """
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # From remaining, split val
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=rel_val, stratify=y_temp, random_state=random_state
    )

    print("=== Stratified split (train/val/test) ===")
    print(f"Train size: {X_train.shape[0]}")
    print(f"Val   size: {X_val.shape[0]}")
    print(f"Test  size: {X_test.shape[0]}")

    # Standardization 标准化
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    return (X_train_sc, X_val_sc, X_test_sc), (y_train, y_val, y_test), scaler


def plot_split_distribution(y_train, y_val, y_test, out_path):
    """
    可视化 train/val/test 每个类别的样本数分布。
    """
    n_classes = len(np.unique(np.concatenate([y_train, y_val, y_test])))
    bins = np.arange(n_classes + 1) - 0.5

    plt.figure(figsize=(10, 4))
    plt.hist(y_train, bins=bins, alpha=0.6, label="train")
    plt.hist(y_val, bins=bins, alpha=0.6, label="val")
    plt.hist(y_test, bins=bins, alpha=0.6, label="test")
    plt.xlabel("Person ID 类别编号")
    plt.ylabel("Count 样本数")
    plt.title("Class distribution in train/val/test\n各子集类别分布")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[SAVE] split distribution -> {out_path}")


# ----------------------------------------------------------
# 3. Dimensionality Reduction: PCA + Autoencoder
# ----------------------------------------------------------
def run_pca(X_train_sc, X_val_sc, X_test_sc, n_components_list=(10, 20, 50, 100)):
    """
    对不同主成分个数做 PCA，并绘制解释方差曲线。
    """
    D = X_train_sc.shape[1]
    max_comp = min(max(n_components_list), D)

    pca = PCA(n_components=max_comp, random_state=42)
    pca.fit(X_train_sc)

    evr = pca.explained_variance_ratio_
    cum_evr = np.cumsum(evr)

    # Plot cumulative explained variance
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cum_evr) + 1), cum_evr, marker="o")
    plt.xlabel("Number of components 主成分个数")
    plt.ylabel("Cumulative explained variance 累积解释率")
    plt.title("PCA cumulative explained variance\nPCA 累积方差解释率")
    plt.grid(True)
    pca_curve_path = os.path.join(OUT_DIR, "pca_cumulative_evr.png")
    plt.tight_layout()
    plt.savefig(pca_curve_path, dpi=150)
    plt.close()
    print(f"[SAVE] PCA cumulative EVR -> {pca_curve_path}")

    # Choose one dimension for downstream use, e.g. 50
    n_pca_dim = min(50, max_comp)
    pca_main = PCA(n_components=n_pca_dim, random_state=42)
    Ztr = pca_main.fit_transform(X_train_sc)
    Zva = pca_main.transform(X_val_sc)
    Zte = pca_main.transform(X_test_sc)

    print(f"[INFO] PCA main dimension = {n_pca_dim}")
    print(f"Train PCA shape: {Ztr.shape}")

    return (Ztr, Zva, Zte), pca_main


def build_autoencoder(input_dim, latent_dim=64):
    """
    Build simple dense autoencoder.
    简单全连接自编码器。
    """
    encoder_inputs = keras.Input(shape=(input_dim,), name="encoder_input")
    x = layers.Dense(256, activation="relu")(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    x = layers.Dense(128, activation="relu")(latent)
    x = layers.Dense(256, activation="relu")(x)
    decoder_outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = keras.Model(encoder_inputs, decoder_outputs, name="autoencoder")
    encoder = keras.Model(encoder_inputs, latent, name="encoder")

    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    return autoencoder, encoder


def run_autoencoder(X_train_sc, X_val_sc, X_test_sc, latent_dim=64, epochs=40, batch_size=64):
    """
    Train autoencoder and return latent representations.
    训练自编码器并返回隐空间表示。
    """
    input_dim = X_train_sc.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, latent_dim)

    history = autoencoder.fit(
        X_train_sc,
        X_train_sc,
        validation_data=(X_val_sc, X_val_sc),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    # Plot AE training curves
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder training curves\n自编码器训练曲线")
    plt.legend()
    ae_curve_path = os.path.join(OUT_DIR, "autoencoder_training_loss.png")
    plt.tight_layout()
    plt.savefig(ae_curve_path, dpi=150)
    plt.close()
    print(f"[SAVE] AE training curves -> {ae_curve_path}")

    Ztr = encoder.predict(X_train_sc)
    Zva = encoder.predict(X_val_sc)
    Zte = encoder.predict(X_test_sc)

    print(f"[INFO] AE latent dim = {latent_dim}, train shape = {Ztr.shape}")
    return (Ztr, Zva, Zte), encoder, autoencoder


# ----------------------------------------------------------
# 4. Clustering on reduced features
# ----------------------------------------------------------
def evaluate_clustering(Z, labels, cluster_labels, method_name):
    """
    Evaluate clustering with ARI / NMI / silhouette and purity.
    用 ARI / NMI / 轮廓系数 + 纯度评估聚类结果。
    """
    ari = adjusted_rand_score(labels, cluster_labels)
    nmi = normalized_mutual_info_score(labels, cluster_labels)

    # silhouette: requires >= 2 clusters and >= 2 samples each
    try:
        sil = silhouette_score(Z, cluster_labels)
    except Exception:
        sil = np.nan

    # purity
    cm = confusion_matrix(labels, cluster_labels)
    purity = np.sum(np.max(cm, axis=0)) / np.sum(cm)

    print(f"\n=== Clustering evaluation ({method_name}) ===")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"Normalized Mutual Info (NMI): {nmi:.4f}")
    print(f"Silhouette score: {sil:.4f}")
    print(f"Cluster purity: {purity:.4f}")


def run_clustering(Z_pca_tr, Z_pca_te, y_train, y_test, n_clusters, prefix="pca"):
    """
    在 PCA 特征上执行 KMeans 和层次聚类，并做 2D 可视化。
    """
    # For visualization, project to 2D PCA
    pca2 = PCA(n_components=2, random_state=42)
    Z2 = pca2.fit_transform(Z_pca_tr)

    # ---------- KMeans ----------
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    k_tr = kmeans.fit_predict(Z_pca_tr)
    k_te = kmeans.predict(Z_pca_te)

    evaluate_clustering(Z_pca_tr, y_train, k_tr, f"{prefix}-KMeans")

    # scatter
    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=k_tr, cmap="tab20", s=10)
    plt.title(f"{prefix.upper()} features - KMeans clustering\n{prefix} 特征 - KMeans 聚类")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    km_path = os.path.join(OUT_DIR, f"{prefix}_kmeans_scatter.png")
    plt.tight_layout()
    plt.savefig(km_path, dpi=150)
    plt.close()
    print(f"[SAVE] {prefix} KMeans scatter -> {km_path}")

    # ---------- Agglomerative ----------
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    a_tr = agg.fit_predict(Z_pca_tr)
    # no predict() for Agglomerative; we'll just assign test later if needed

    evaluate_clustering(Z_pca_tr, y_train, a_tr, f"{prefix}-Agglomerative")

    plt.figure(figsize=(6, 5))
    plt.scatter(Z2[:, 0], Z2[:, 1], c=a_tr, cmap="tab20", s=10)
    plt.title(
        f"{prefix.upper()} features - Agglomerative clustering\n{prefix} 特征 - 层次聚类 (ward)"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    ag_path = os.path.join(OUT_DIR, f"{prefix}_agg_scatter.png")
    plt.tight_layout()
    plt.savefig(ag_path, dpi=150)
    plt.close()
    print(f"[SAVE] {prefix} Agglomerative scatter -> {ag_path}")

    return kmeans, k_tr, k_te


# ----------------------------------------------------------
# 5. Supervised NN classifier
# ----------------------------------------------------------
def build_classifier(input_dim, n_classes):
    """
    Build a simple feed-forward NN classifier.
    建立一个简单的前馈神经网络分类器。
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    images_test,
    n_classes,
    feature_name="PCA",
):
    """
    用给定特征训练分类器，并在 val/test 上评估。
    """
    input_dim = X_train.shape[1]
    model = build_classifier(input_dim, n_classes)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=64,
        callbacks=callbacks,
        verbose=2,
    )

    # Plot training curves
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy 准确率")
    plt.title(f"{feature_name} NN training accuracy\n{feature_name} 特征神经网络训练准确率")
    plt.legend()
    acc_path = os.path.join(OUT_DIR, f"{feature_name.lower()}_nn_accuracy.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[SAVE] NN training acc -> {acc_path}")

    # Evaluate on test
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n=== NN Test Performance ({feature_name}) ===")
    print(f"Test accuracy: {test_acc:.4f}, loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print("\nClassification report on test:")
    print(classification_report(y_test, y_pred))

    # Show some sample predictions
    num_show = 8
    idx = np.random.choice(len(y_test), size=num_show, replace=False)

    plt.figure(figsize=(10, 4))
    for i, j in enumerate(idx):
        img = images_test[j]
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"True:{y_test[j]} Pred:{y_pred[j]}")
    plt.suptitle(
        f"Sample test images: true vs predicted ({feature_name})\n测试样本: 真实标签 vs 预测标签"
    )
    img_path = os.path.join(OUT_DIR, f"{feature_name.lower()}_test_examples.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.close()
    print(f"[SAVE] sample predictions -> {img_path}")

    return model


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    # ---------- Step 1: Load UMIST ----------
    print("=== Step 1: Load UMIST dataset ===")
    images, labels, df = load_umist("umist_cropped.mat")
    N, H, W = images.shape
    X = images.reshape(N, -1)

    # ---------- Step 2: Stratified split + scaling ----------
    print("\n=== Step 2: Stratified split & scaling ===")
    (X_train_sc, X_val_sc, X_test_sc), (y_train, y_val, y_test), scaler = (
        stratified_split_and_scale(X, labels)
    )

    # For visualizing test images later
    # 把 images 拆分为与特征一致的顺序
    X_train, X_test_tmp, img_train, img_test_tmp, y_train_raw, y_test_raw = train_test_split(
        X, images, labels, test_size=0.2, stratify=labels, random_state=42
    )
    rel_val = 0.2 / 0.8
    X_train_tmp, X_val_tmp, img_train_tmp, img_val_tmp, y_train_tmp, y_val_tmp = (
        train_test_split(
            X_train, img_train, y_train_raw, test_size=rel_val, stratify=y_train_raw, random_state=42
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