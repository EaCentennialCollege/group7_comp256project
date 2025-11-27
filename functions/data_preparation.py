import numpy as np
from scipy.io import loadmat
import pandas as pd


# ----------------------------------------------------------
# 1. Load UMIST .mat helper
# ----------------------------------------------------------
def load_umist(mat_path="datasets/umist_cropped.mat"):
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
