import sys
from .Backend_src.DecisionTreeCreator import DecisionTreeCreator
import os
import pickle
import numpy as np
class WeightedDecisionTrees():
    def __init__(self, config):
        self.config = config
        self.tree_dict = None
        self.tree = None
    
    def fit(self, X, Y, FI=None, out_logits = None):
        self.DecisionTree = DecisionTreeCreator(self.config["Tree"])
        self.DecisionTree.fit(X, Y, FI, out_logits)
        self.tree_dict = self.DecisionTree.tree
    
    def _forward(self, x):
        if self.tree is not None:
            return self.tree._forward(x)
        elif self.tree_dict is not None:
            return self._dictionary_forward(x)

    def _dictionary_forward(self, x):
        value = None
        current_dict = self.tree_dict
        while value == None:
            if None in current_dict.keys():
                current_dict = current_dict[None]
                continue
            if "value" in current_dict.keys():
                return current_dict["value"]
            
            if x[current_dict["feature"]] <= current_dict["threshold"]:
                current_dict = current_dict["left"]
            elif x[current_dict["feature"]] > current_dict["threshold"]:
                current_dict = current_dict["right"]
            else:
                break
        return value

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if self.tree_dict != None:
            with open(filepath, "wb")as file :
                pickle.dump(self.tree_dict, file)
    
    def load(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "rb") as f:
            tree_dict = pickle.load(f)
            self.tree_dict = tree_dict

    def Predict(self, X):
        return np.stack([self._forward(x) for x in X])
    
    def act(self, x):
        return self._forward(x)


