import pandas as pd
import numpy as np
from .Node import Single_Attribute_Node
class DecisionTree():
    def __init__(self, config):
        self.config = config
    
    def fit(self, X,Y):
        print(len(Y))
        X = pd.DataFrame(X, columns=self.config["picture"]["labels"])
        if self.config["surrogate"]["classifier"]:
            Y = pd.DataFrame(Y, columns=["out"])
        else:
            Y = pd.DataFrame(Y, columns=self.config["picture"]["class_names"])
        self.root = Single_Attribute_Node(self.config)
        self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0)

    def _forward(self, val):
        return self.root._forward(val)
    
    def predict(self, vals):
        X = pd.DataFrame(vals, columns=self.config["picture"]["labels"])

        results = []
        for index, row in X.iterrows():
            results.append(self._forward(row))

        return results
    def get_dict_representation(self):
        return self.dictionary_rep
    def printer(self):
        print("root")
        print(self.root.printer())
    
