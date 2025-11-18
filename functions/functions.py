import matplotlib.pyplot as plt
import pandas as pd

def plot_class_counts(labels, title):
    counts = pd.Series(labels).value_counts().sort_index()
    counts.plot(kind="bar")
    plt.xlabel("Person ID")
    plt.ylabel("Number of images")
    plt.title(title)
    plt.tight_layout()
    plt.show()
