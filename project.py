import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import functions.functions as fn

mat = sio.loadmat("umist_cropped.mat", squeeze_me=True)

facedat = mat["facedat"]  # object array, one entry per person
dirnames = mat["dirnames"]  # labels (one per person)

X_list = []
y_list = []

for person_idx, person_imgs in enumerate(facedat):
    # use directory name as label (or just person_idx)
    label = str(dirnames[person_idx])

    if person_imgs.ndim == 3:
        # if a stack of 3d images
        imgs = np.moveaxis(person_imgs, -1, 0)  # -> (num_imgs, H, W)
    else:
        # if a stack of 2d images
        imgs = person_imgs

    for img in imgs:
        # convert to float32 and flatten
        img = np.asarray(img, dtype=np.float32)
        # normalize to [0, 1]
        X_list.append(img.flatten())
        # append label
        y_list.append(label)

# stack all data
X = np.vstack(X_list)
y = np.array(y_list)

# create DataFrame
pixel_cols = [f"px_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=pixel_cols)
df["label"] = y

print("\nDataFrame shape: ", df.shape)
print("DataFrame head: \n", df.head())


# Data Splitting Start!
X = df.drop(columns=["label"]).values
y = df["label"].values



# First split: 
# train (70%)
# temp (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Second split: 
# validation (15%)
# test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)


print("\nTrain : ", X_train.shape)
print("Validation : ", X_val.shape)
print("Test : ", X_test.shape)


# Normalize (fit only on train to avoid data leakage)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)


fn.plot_class_counts(y_train, "Training set class distribution")
fn.plot_class_counts(y_val,   "Validation set class distribution")
fn.plot_class_counts(y_test,  "Test set class distribution")

fig, axes = plt.subplots(2, 5, figsize=(15, 3))
fig.suptitle("Sample Images")

for i, ax in enumerate(axes.flatten()):
    img = X_train[i].reshape(112, 92)
    label = y_train[i]

    # Display the image
    ax.imshow(img, cmap="gray")
    ax.set_title(f"Label: {label}")
    ax.axis("off")

plt.tight_layout()
plt.show()
