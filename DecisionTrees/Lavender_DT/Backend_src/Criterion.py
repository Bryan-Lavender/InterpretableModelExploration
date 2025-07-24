import numpy as np


#discrete variables
def entropy(Y):
    if Y.size <= 1:
        return 0.0

    _, counts = np.unique(Y, return_counts=True)
    probabilities = counts / Y.size
    return -np.sum(probabilities * np.log2(probabilities))

def MSE(Y):
    if len(Y) == 1:
        return 0
    ym = (1/len(Y)) * Y.sum(axis=0)
    return (1/len(Y)) * np.linalg.norm(Y-ym, axis=1).sum()

