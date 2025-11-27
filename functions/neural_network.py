import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
import numpy as np
from sklearn.metrics import classification_report
import os
from utils.constants import OUT_DIR


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
    plt.title(
        f"{feature_name} NN training accuracy\n{feature_name} 特征神经网络训练准确率"
    )
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
