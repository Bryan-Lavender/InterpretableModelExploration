import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle
import os

from .weighingmethods import Q_diff
class VIPER_weighted():
    def __init__(self, config):
        self.config = config
        self.tree = None

    def train_viper_categorical(self,X,Y, Activations):
        weights = Q_diff(Activations)
     
        clf = DecisionTreeClassifier(max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0,
            criterion="entropy")
        self.tree = clf.fit(X, Y, sample_weight=weights)

    def _forward(self, x):
        
        return self.tree.predict(np.array([x]))

    def Predict(self, X):
        return self.tree.predict(X)

    def act(self, X):
        return self.tree.predict(np.array([X]))[0]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tree, f)
    def load(self, path):
        with open(path, "rb") as f:
            self.tree = pickle.load(f)
    
class VIPER_reSampled():
    def __init__(self, config):
        self.config = config
        self.tree = None

    def train_viper_categorical(self,X,Y, Activations):
        weights = Q_diff(Activations)
        X,Y = self.stochastic_filter_by_weight(X,Y,weights)
        clf = DecisionTreeClassifier(max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            ccp_alpha=0.0,
            criterion="entropy")
        self.tree = clf.fit(X, Y)
    
    def _forward(self, x):
        
        return self.tree.predict(np.array([x]))

    def Predict(self, X):
        return self.tree.predict(X)

    def act(self, X):
        return self.tree.predict(np.array([X]))[0]
    
    def stochastic_filter_by_weight(self, X, Y, Weights):
        weights = np.asarray(Weights, dtype=float)

        norm_weights = (weights - weights.min()) / (weights.max() - weights.min())

        mask = np.random.rand(len(norm_weights)) < norm_weights

        X_filtered = X[mask]
        Y_filtered = Y[mask]
        indices = np.nonzero(mask)[0]

        return X_filtered, Y_filtered, indices
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tree, f)
    def load(self, path):
        with open(path, "rb") as f:
            self.tree = pickle.load(f)