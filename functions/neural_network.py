import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
from sklearn.metrics import classification_report
import os
from utils.constants import OUT_DIR
import tensorflow as tf
from tensorflow.keras import layers, models


# ----------------------------------------------------------
# 5. Supervised NN classifier
# ----------------------------------------------------------

# CNN model
def build_cnn(input_shape, n_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(n_classes, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# Train cnn classifier
def train_cnn_classifier(
    X_train_img, y_train,
    X_val_img, y_val,
    X_test_img, y_test,
    n_classes
):
    

    # Normalize
    X_train_img = X_train_img.astype("float32") / 255.0
    X_val_img   = X_val_img.astype("float32") / 255.0
    X_test_img  = X_test_img.astype("float32") / 255.0

    # Add channel dimension
    if X_train_img.ndim == 3:
        X_train_img = np.expand_dims(X_train_img, axis=-1)
        X_val_img   = np.expand_dims(X_val_img, axis=-1)
        X_test_img  = np.expand_dims(X_test_img, axis=-1)

    input_shape = X_train_img.shape[1:]

    # Build CNN
    model = build_cnn(input_shape, n_classes)

    # Early stopping
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True
        )
    ]

    # Training
    history = model.fit(
        X_train_img, y_train,
        validation_data=(X_val_img, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=2,
    )

    # Plot training curves
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy ")
    plt.title("CNN training accuracy")
    plt.legend()
    acc_path = os.path.join(OUT_DIR, "cnn_training_accuracy.png")
    plt.tight_layout()
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"[SAVE] CNN training accuracy -> {acc_path}")

    # Evaluate on test
    test_loss, test_acc = model.evaluate(X_test_img, y_test, verbose=0)
    print("\n=== CNN Test Performance ===")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    # Predictions
    y_pred = np.argmax(model.predict(X_test_img), axis=1)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))




    # Show some sample predictions
    num_show = 8
    idx = np.random.choice(len(y_test), size=num_show, replace=False)

    plt.figure(figsize=(10, 4))
    for i, j in enumerate(idx):
        img = X_test_img[j].squeeze()
        plt.subplot(2, 4, i + 1)
        plt.imshow(img, cmap="gray")
        plt.axis("off")
        plt.title(f"True:{y_test[j]} Pred:{y_pred[j]}")
        
    plt.suptitle("CNN – Sample Test Predictions")
    img_path = os.path.join(OUT_DIR, "cnn_test_examples.png")
    plt.tight_layout()
    plt.savefig(img_path, dpi=150)
    plt.show()
    plt.close()
    print(f"[SAVE] CNN sample predictions → {img_path}")

    return model, history
