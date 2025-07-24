import pandas as pd
import numpy as np



def STD_diff(y_values, threshold_factor=2):

    if len(y_values) == 1:
        return y_values[0].astype(float)

    mean_value = np.mean(y_values, axis=0)
    distances = np.linalg.norm(y_values - mean_value, axis=1)
    std_dev = np.std(distances)

    threshold = threshold_factor * std_dev * np.sqrt(y_values.shape[1])

    if np.all(distances < threshold):
        return mean_value.astype(float)

    return None  

def one_class_left(Y):
    if len(Y) == 1:
        return Y[0]
    unique = np.unique(Y)
    if unique.size == 1:
        return int(unique[0])
    return None