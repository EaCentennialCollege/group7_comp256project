# modified by Erwin Julian Alapide
# Step 2: Data Splitting

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def stratified_split_and_scale(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Splits data into stratified train, validation, and test sets, then scales the features.

    This function first splits the data into a temporary training set and a test set.
    Then, it splits the temporary training set into the final training and validation sets.
    This two-step process ensures the correct proportions for all three datasets.
    Stratification is used to maintain the same percentage of samples for each class
    in all subsets.

    Args:
        X (np.ndarray): The feature matrix (e.g., flattened images).
        y (np.ndarray): The target labels.
        test_size (float): The proportion of the dataset to include in the test split.
        val_size (float): The proportion of the dataset to include in the validation split.
        random_state (int): A seed for the random number generator for reproducibility.

    Returns:
        tuple: A tuple containing (X_train_sc, X_val_sc, X_test_sc).
        tuple: A tuple containing (y_train, y_val, y_test).
        StandardScaler: The scaler object fitted on the training data.
    """
    # Step 1: Split the data into a temporary training set and a test set.
    # The `stratify=y` argument ensures both sets have similar class distributions.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Step 2: Split the temporary training set into the final training and validation sets.
    # The validation size needs to be recalculated as a proportion of the temporary set.
    remaining_size = 1.0 - test_size
    relative_val_size = val_size / remaining_size

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=random_state
    )

    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Validation set size: {X_val.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    # Step 3: Scale the feature data.
    # It's crucial to fit the scaler ONLY on the training data to avoid data leakage.
    # Then, use the fitted scaler to transform the train, validation, and test sets.
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)

    # Package the results into tuples for easy handling
    X_splits = (X_train_sc, X_val_sc, X_test_sc)
    y_splits = (y_train, y_val, y_test)

    return X_splits, y_splits, scaler


def plot_split_distribution(y_train, y_val, y_test, output_path):
    """
    Visualizes and saves the class distribution across train, validation, and test sets.

    This helps to verify that the stratified split was successful and that each
    dataset has a representative sample of all classes.

    Args:
        y_train (np.ndarray): Training set labels.
        y_val (np.ndarray): Validation set labels.
        y_test (np.ndarray): Test set labels.
        output_path (str): The file path to save the plot image.
    """
    # Determine the number of unique classes to set up the histogram bins
    all_labels = np.concatenate([y_train, y_val, y_test])
    num_classes = len(np.unique(all_labels))
    # Create bins centered around each class integer label
    bins = np.arange(num_classes + 1) - 0.5

    plt.figure(figsize=(10, 5))

    # Plot a histogram for each data split
    plt.hist(y_train, bins=bins, alpha=0.7, label="Train Set", rwidth=0.8)
    plt.hist(y_val, bins=bins, alpha=0.7, label="Validation Set", rwidth=0.8)
    plt.hist(y_test, bins=bins, alpha=0.7, label="Test Set", rwidth=0.8)

    plt.xlabel("Class ID")
    plt.ylabel("Number of Samples")
    plt.title("Class Distribution in Train, Validation, and Test Sets")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the figure to the specified path
    plt.savefig(output_path, dpi=150)
    plt.close()
