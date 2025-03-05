import numpy as np
import pandas as pd

class Node():

    def __init__(self):
        self.feature_index = None
        self.is_leaf = None
        self.val_bucket = None
        self.return_value = None
        self.parent_node = None
        self.represented_nodes = None
        self.depth = None
        
    def _forward(self, Val):
        if self.is_leaf:
            return self.return_value
        
        if type(self.val_bucket) == float or type(self.val_bucket) == np.float64:
            if Val[self.feature_index] <= self.val_bucket:
                return self.left_node._forward(Val)
            else:
                return self.right_node._forward(Val)
        else:
            if Val[self.feature_index] == self.val_bucket:
                return self.left_node._forward(Val)
            else:
                return self.right_node._forward(Val)