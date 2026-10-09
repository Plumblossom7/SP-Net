import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_confuse_matrix(cm, classes, root_folder, cv):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.savefig(f'{root_folder}/conf/fold_{cv}.png', dpi=600)