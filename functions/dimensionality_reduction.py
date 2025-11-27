from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras import layers
from utils.constants import OUT_DIR


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


def run_autoencoder(
    X_train_sc, X_val_sc, X_test_sc, latent_dim=64, epochs=40, batch_size=64
):
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
