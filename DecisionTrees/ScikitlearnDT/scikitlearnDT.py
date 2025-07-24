import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pickle
import os

class SKLTree():
    def __init__(self, config):
        self.config = config
        self.tree = None

    def fit(self,X,Y):
    
        if self.config["surrogate"]["classifier"]:
            clf = DecisionTreeClassifier(max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    ccp_alpha=0.0, criterion="entropy")
            self.tree = clf.fit(X, Y)
        else:
            clf = DecisionTreeRegressor(max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    ccp_alpha=0.0, criterion="squared_error")
            self.tree = clf.fit(X, Y)
            
    def _forward(self, x):
        
        return self.tree.predict(np.array([x]))

    def Predict(self, X):
        return self.tree.predict(X)
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tree, f)
    def load(self, path):
        with open(path, "rb") as f:
            self.tree = pickle.load(f)



    