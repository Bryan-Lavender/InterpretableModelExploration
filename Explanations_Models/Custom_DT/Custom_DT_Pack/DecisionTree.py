import pandas as pd
import numpy as np
from .Node import Single_Attribute_Node
class DecisionTree():
    def __init__(self, config, FI=None):
        self.config = config
        self.FI= FI
    
    def fit(self, X,Y):
        
        X = pd.DataFrame(X, columns=self.config["picture"]["labels"])
        if self.config["surrogate"]["classifier"]:
            Y = pd.DataFrame(Y, columns=["out"])
        else:
            Y = pd.DataFrame(Y, columns=self.config["picture"]["class_names"])
        self.root = Single_Attribute_Node(self.config)
        if self.config["surrogate"]["use_FI"]:
            out, FI = self.FI.Relevence(X.to_numpy())
            FI = np.abs(FI)
            columns = pd.MultiIndex.from_product([self.config["picture"]["class_names"], self.config["picture"]["labels"]], names=['OutLogit', 'InLogit'])

            # Reshape the data to (60, 8) to match the multi-level column structure
            reshaped_data = FI.reshape(FI.shape[0], -1)

            # Create the DataFrame
            FI = pd.DataFrame(reshaped_data, columns=columns)
            out = pd.DataFrame(out, columns=self.config["picture"]["class_names"])
            self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0, FI = FI, out_logits = out)
        else:    
            self.dictionary_rep, self.max_depth = self.root.fit(X,Y,0)

        self.nodeList = self.node_list()

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
    
    def TraverseTree(self,node:Single_Attribute_Node):
        if node.is_leaf:
            return [node]
        else:
            re = []
            left = self.TraverseTree(node.left_node)
            right = self.TraverseTree(node.right_node)
            for i in left:
                re.append(i)
            for i in right:
                re.append(i)
            return re
    
    def node_list(self):
        return self.TraverseTree(self.root)
    
    def get_node_list(self):
        return self.nodeList
    
    def get_depth(self):
        return self.max_depth
    
    def get_breadth(self):
        breadths = [0 for i in range(self.max_depth + 1)]
        for node in self.nodeList:
            breadths[node.depth] += 1
        return max(breadths)

    def get_avg_representation(self):
        Running_Avg = 0
        for node in self.nodeList:
            Running_Avg += node.represented_nodes
        return Running_Avg/len(self.nodeList)
    
    def get_metrics(self):
        return {"Representation": self.get_avg_representation(), "Depth": self.get_depth(), "Breadth": self.get_breadth()}
    
    
        
