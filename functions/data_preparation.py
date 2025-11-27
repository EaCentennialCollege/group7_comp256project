# modified by Erwin Julian Alapide
# Step 1: Data Preparation

import numpy as np
from scipy.io import loadmat
import pandas as pd


# ----------------------------------------------------------
# 1. Load UMIST .mat helper
# ----------------------------------------------------------
def load_umist(mat_path="datasets/umist_cropped.mat"):

    # Load the .mat file, which is a dictionary-like object
    try:
        mat = loadmat(mat_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at: {mat_path}")

    # Check if the required 'facedat' key exists in the file
    if "facedat" not in mat:
        raise ValueError(
            f"'facedat' key not found in {mat_path}. Available keys: {list(mat.keys())}"
        )

    # 'facedat' is a 2D numpy array where each element is another array of images.
    # It's structured like [[person1_images, person2_images, ...]]
    facedat = mat["facedat"]
    num_persons = facedat.shape[1]

    # Lists to store the processed images and their corresponding labels
    all_images = []
    all_labels = []

    # The images for the first person are used to determine the height (H) and width (W)
    # This assumes all images in the dataset have the same dimensions.
    first_person_images = facedat[0, 0]
    if first_person_images.ndim == 3:
        # Case 1: Images are in a 3D array of shape (Height, Width, NumImages)
        H, W, _ = first_person_images.shape
    elif first_person_images.ndim == 2:
        # Case 2: Images are flattened into a 2D array of shape (Height*Width, NumImages)
        # We assume the standard UMIST image size of 112x92
        H, W = 112, 92
        if first_person_images.shape[0] != H * W:
            # If the size doesn't match, raise an error for clarity.
            raise ValueError(
                f"Unexpected image format. Expected flattened {H*W} pixels."
            )
    else:
        raise ValueError("Unexpected array dimension for image data.")

    print(f"[INFO] Inferred image size: {H}x{W} pixels.")

    # Loop through each person in the dataset
    for person_id in range(num_persons):
        person_images_data = facedat[0, person_id]
        num_images_for_person = person_images_data.shape[-1]

        # Loop through all images for the current person
        for image_index in range(num_images_for_person):
            if person_images_data.ndim == 3:
                # If data is (H, W, num_images), extract the image slice
                img = person_images_data[:, :, image_index]
            else:  # ndim == 2
                # If data is (H*W, num_images), extract the column and reshape it
                img = person_images_data[:, image_index].reshape(H, W)

            # Add the processed image and its label to our lists
            all_images.append(img.astype(np.float32))
            all_labels.append(person_id)

    # Convert the lists of images and labels into NumPy arrays
    images = np.stack(all_images)  # Shape: (Total_Images, H, W)
    labels = np.array(all_labels, dtype=np.int32)  # Shape: (Total_Images,)
    total_images = len(images)

    # For machine learning, flatten each image from a 2D matrix (H, W)
    # to a 1D vector (H * W).
    flattened_images = images.reshape(total_images, H * W)

    # Create a Pandas DataFrame for easier data manipulation
    df = pd.DataFrame(flattened_images)
    df["person_id"] = labels

    print(f"[INFO] Loaded {total_images} images from {num_persons} persons.")
    return images, labels, df
