import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


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
