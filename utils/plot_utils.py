import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from utils.path_utils import join_path
from utils.label_dicts import *



def compute_class_frequencies(labels, num_classes):
    frequencies = np.zeros(num_classes, dtype=float)
    for label in labels:
        frequencies[label] += 1
    frequencies /= np.sum(frequencies)
    return frequencies


def compute_class_frequencies_no_direction(labels, num_classes):
    frequencies = np.zeros(num_classes)
    for label in labels:
        if label == 0:
            index = label
        else:
            index = math.ceil(label / 2)
        frequencies[index] += 1
    frequencies /= np.sum(frequencies)
    return frequencies


def plot_classes_histogram(labels, save_path, partition='whole'):
    sns.set(style="whitegrid")
    num_classes = len(label2class)
    frequencies = compute_class_frequencies(labels, num_classes)
    class_names = list(class2label.keys())
    
    plt.figure(figsize=(9, 6))
    plt.bar(range(num_classes), frequencies, label='Class proportion', log=True)
    plt.xticks([i for i in range(num_classes)], class_names, rotation='vertical')
    plt.legend()
    plt.title(f'Classes proportion in the {partition} dataset considering directionality')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{partition}_class_proportion.png'))
    plt.close()


def plot_classes_histogram_no_direction(labels, save_path, partition='whole'):
    sns.set(style="whitegrid")
    num_classes = len(class_names_no_direction)
    frequencies = compute_class_frequencies_no_direction(labels, num_classes)
    class_names = class_names_no_direction
    
    plt.figure(figsize=(9, 6))
    plt.bar(range(num_classes), frequencies, label='Class proportion', log=True)
    plt.xticks([i for i in range(num_classes)], class_names, rotation='vertical')
    plt.legend()
    plt.title(f'Classes proportion in the {partition} dataset without directionality')
    plt.tight_layout()
    plt.savefig(join_path(save_path, f'{partition}_class_proportion_no_direction.png'))
    plt.close()


def plot_confusion_matrix(values, class_names, save_path, method_name, with_direction=True):
    confusion_matrix_df = pd.DataFrame(values, index=class_names, columns=class_names)
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1.5)
    sns.heatmap(confusion_matrix_df, annot=True, annot_kws={"size": 16}, fmt='g')
    plt.title(f'Confusion matrix using {method_name} on the test dataset')
    plt.ylabel('Ground-Truth')
    plt.xlabel('Predictions')
    plt.tight_layout()

    method_name = method_name.replace(' ', '_')
    if with_direction:
        name = f'test_confusion_matrix_{method_name}.png'
    else:
        name = f'test_confusion_matrix_no_direction_{method_name}.png'

    plt.savefig(join_path(save_path, name), dpi=150)
    plt.close()