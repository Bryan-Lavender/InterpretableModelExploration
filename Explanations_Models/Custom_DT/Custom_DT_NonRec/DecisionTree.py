import pandas as pd
import numpy as np
from .Node import Node
from .BucketAlgs import half_split
from .SplittingFunctions import entropy, MSE, MAE
from scipy.spatial.distance import pdist
from .FI_Calulator import Policy_Smoothing_Max_Weighting, Policy_Smoothing_MaxAvg_Weighting, Policy_Smoothing_Var_Weighting, Classification_Forgiveness
bucketing_mech = {"half_split": half_split}
splitting_functions = {"entropy": entropy, "MSE": MSE, "MAE": MAE}
FICalcs = {"Var_weighted": Policy_Smoothing_Var_Weighting, "Max_all": Policy_Smoothing_Max_Weighting, "Max_avg":Policy_Smoothing_MaxAvg_Weighting, "class": Classification_Forgiveness}

class DecisionTree():
    def __init__(self, config, FI=None):
        self.config = config
        self.FI = FI
        self.single_tree = not config["surrogate"]["multi_tree"]
        self.FI_set = False

    def fit(self, X,Y, FI_in = None, out_logits = None):

        if self.single_tree:
            self.fit_single(X,Y,FI_in = FI_in, out_logits = out_logits)
        else:
            self.fit_multi_tree(X,Y,FI_in = FI_in, out_logits = out_logits)
    
    def fit_single(self,X,Y, FI_in=None, out_logits = None):
        X = pd.DataFrame(X, columns = self.config["picture"]["labels"])
        if self.config["surrogate"]["classifier"]:
            Y = pd.DataFrame(Y, columns=["out"])
        else:
            Y = pd.DataFrame(Y, columns = self.config["picture"]["class_names"])

        stack = []
        self.root = Node()
        stack.append([self.root, X, Y, FI_in, out_logits])
        
        self.max_depth = -1
        while stack:
            node, X, Y, FI_in, out_logits = stack.pop()
            left_right = self.fit_node(node, X, Y, node.depth, FI_in, out_logits)
            

            #test if tree is leaf, if so continue
            if node.is_leaf == True:
                depth = left_right[1]
                if depth > self.max_depth:
                    self.max_depth = depth
                continue


            #append necessary values
            stack.append([node.left_node] + left_right["left"])
            stack.append([node.right_node] + left_right["right"])
        self.nodeList = self.node_list()
        self.dictionary_rep = self.tree_to_dict()
    
    def should_create_leaf(self, y_values, threshold_factor=2):
        # Calculate the mean of the values
        mean_value = np.mean(y_values)
        
        # Calculate the Euclidean distances from the mean
        distances = np.linalg.norm(y_values - mean_value, axis=1)
        
        # Calculate the standard deviation of the distances
        std_dev = np.std(distances)
        
        # Set the threshold based on standard deviation and vector size
        
        threshold = threshold_factor * std_dev * np.sqrt(len(y_values[0]))
        
        # Check if all distances are below the threshold
        if np.all(distances < threshold):
            return True  # Create a leaf node
        else:
            return False  # Continue splitting
        
    def fit_node(self, node, X,Y, depth = None, FI = None, out_logits = None):
        if depth != None:
            node.depth = depth + 1
        else:
            node.depth = 1
        if self.config["surrogate"]["classifier"] and len(Y[Y.keys()[0]].unique()) == 1:
            node.represented_nodes = len(Y)
            node.return_value = Y[Y.keys()[0]].value_counts().index[0]
            node.is_leaf = True

            node.left_node = -1
            node.right_node = -1
            return {"Value": int(node.return_value)},self.depth
        
        
        elif not self.config["surrogate"]["classifier"] and (len(Y) == 1 or self.should_create_leaf(Y.to_numpy())):
            if len(Y) == 1:
                node.return_value = Y.to_numpy()[0]
                node.represented_nodes = len(Y)
            else:
                node.represented_nodes = len(Y)
                node.return_value = np.mean(Y.to_numpy(), axis=0)
                

            node.is_leaf = True

            node.left_node = -1
            node.right_node = -1
            to_return = []
           
            for i in node.return_value:
                to_return.append(float(i))
            return ({"Value": to_return},node.depth)
        else:
            buckets = self.bucket(X)
            
            min_var, min_bucket, min_val, indicies_left, indicies_right = self.split(X,Y, buckets, FI=FI, out_logits=out_logits)
            node.feature_index = min_var
            node.val_bucket = min_bucket
            node.heuristic_value = min_val
            
     
            node.left_node = Node()
            node.right_node = Node()

            #set parent
            node.left_node.parent = node
            node.right_node.parent = node
            if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                return {"left": [X.loc[indicies_left], Y.loc[indicies_left], FI.loc[indicies_left], out_logits.loc[indicies_left]], "right": [X.loc[indicies_right], Y.loc[indicies_right], FI.loc[indicies_right], out_logits.loc[indicies_right]]}

            else:
                return {"left": [X.loc[indicies_left], Y.loc[indicies_left], None, None], "right": [X.loc[indicies_right], Y.loc[indicies_right], None, None]}
                
    

    def split(self, X, Y, buckets, FI = None, out_logits = None):
        #heuristics = {}
        min_var = None
        min_bucket = None
        min_val = None
        
        if self.config["surrogate"]["use_FI"]:
            if self.config["surrogate"]["multi_tree"]:
        
                FI_val = FI.mean()
            else:
                
                FI_val = FICalcs[self.config["FI"]["grouping"]](FI, out_logits)
        
            
        for var in buckets.keys():
                for bucket in buckets[var]:
                    
                    # calculate entropy 
                    if type(bucket) == np.float64 or type(bucket) == np.float32 or type(bucket) == float:
                        indicies_left = X[X[var] <= bucket].index
                    else:
                        indicies_left = X[X[var] == bucket].index
                    
                    indicies_right = X.index.difference(indicies_left)
                    if len(indicies_left) == 0 or len(indicies_right) == 0:
                        continue
                    Y_left = Y.loc[indicies_left]
                    Y_right = Y.loc[indicies_right]
                
                
                    heuristic_left  = 0 if len(indicies_left) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_left)
                    heuristic_right = 0 if len(indicies_right) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_right)
                
                    heuristic_all = splitting_functions[self.config["surrogate"]["criterion"]](Y)
                   
                  
                    val = heuristic_all - (len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right)
                    if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                        #"""DO FOR CONTRIBUTION BASED FI, further research required to make better but I believe it is possible"""
                        if self.config["FI"]["method"] == "LRP":
                            if self.config["surrogate"]["multi_tree"]:
                                FI_left = FI.loc[indicies_left].mean()
                                FI_right =FI.loc[indicies_right].mean()
                                val = FI_left[var]*len(Y_left)/len(Y)*heuristic_left + FI_right[var]*len(Y_right)/len(Y)*heuristic_right
                            else:
                                FI_left = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_left], out_logits.loc[indicies_left])
                                FI_right = FICalcs[self.config["FI"]["grouping"]](FI.loc[indicies_right], out_logits.loc[indicies_right])
                                val = FI_left[var].iloc[0]*len(Y_left)/len(Y)*heuristic_left + FI_right[var].iloc[0]*len(Y_right)/len(Y)*heuristic_right
                        
                        
                        #"""DO FOR SENSITIVITY BASED FI, really is the meat of the papers rn"""
                        elif self.config["FI"]["method"] == "FD":
                            if self.config["surrogate"]["multi_tree"]:
                                e = 0.0000000001
                                FI_all = FI_val[var] + e
                                val = (FI_all)*(heuristic_all - (len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right) )   
                            else:
                                val = (FI_val[var].iloc[0]) * val

                    if self.config["FI"]["method"] == "FD":
                        if min_val == None or min_val < val:
                            min_val = val
                            min_var = var
                            min_bucket = bucket

                            curr_ind_left = indicies_left
                            curr_ind_right = indicies_right
                    elif self.config["FI"]["method"] == "LRP":
                        if min_val == None or min_val > val:
                            min_val = val
                            min_var = var
                            min_bucket = bucket

                            curr_ind_left = indicies_left
                            curr_ind_right = indicies_right
                    
            #else:

        return (min_var, min_bucket, min_val, curr_ind_left, curr_ind_right)
    
    def bucket(self, X, feature_types = None):
        buckets = {}
        if feature_types == None or self.config == None:
            for key in X.keys():
                if X[key].dtype == "float64" or X[key].dtype == "float32":
                    buckets[key] = bucketing_mech[self.config["surrogate"]["bucket_alg"]](X[key].unique())
                else:
                    buckets[key] = list(X[key].unique())
        else:
            for key in X.keys():
                if self.config["feature_types"][key] == "cont":
                    buckets[key] = bucketing_mech[self.config["surrogate"]["bucket_alg"]](X[key])
                else:
                    buckets[key] = list(X[key].unique())
     
        return buckets
    
    def tree_to_dict(self):
        if not self.root:
            return {}

        stack = [(self.root, {})]  # Stack stores (node, corresponding dictionary)
        tree_dict = {}

        while stack:
            node, node_dict = stack.pop()

            # Assign node attributes
            node_dict["Feature"] = node.feature_index
            node_dict["Value"] = node.val_bucket

            # Process left and right children
            if node.left_node != -1:
                node_dict["Left_Child"] = {}
                stack.append((node.left_node, node_dict["Left_Child"]))
            else:
                node_dict["Left_Child"] = None

            if node.right_node != -1:
                node_dict["Right_Child"] = {}
                stack.append((node.right_node, node_dict["Right_Child"]))
            else:
                node_dict["Right_Child"] = None
            if node.is_leaf:
                node_dict["Output"] = list(np.array(node.return_value, dtype=float))
            # If this is the root node, store it in the return dictionary
            if node == self.root:
                tree_dict = node_dict

        return tree_dict

    def get_FI(self,X):
        out, FI = self.FI.Relevence(X)
        FI = np.abs(FI)
        columns = pd.MultiIndex.from_product([self.config["picture"]["class_names"], self.config["picture"]["labels"]], names=['OutLogit', 'InLogit'])

        # Reshape the data to (60, 8) to match the multi-level column structure
        reshaped_data = FI.reshape(FI.shape[0], -1)

        # Create the DataFrame
        FI = pd.DataFrame(reshaped_data, columns=columns)
        out = pd.DataFrame(out, columns=self.config["picture"]["class_names"])
        return out,FI
        
    def _forward(self, val):
        if self.single_tree:
            return self.root._forward(val)
        else:
            re = []
            for i in self.config["picture"]["class_names"]:
                re.append(self.roots[i]._forward(val)[0])
            re = np.array(re)
            if self.config["env"]["discrete"]:
                
                re = np.argmax(re)
                
            return re
    
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
    
    def TraverseTree(self,node):
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
        if self.config["surrogate"]["multi_tree"]:
            nodes = {}
            for i in self.config["picture"]["class_names"]:
                nodes[i] = self.TraverseTree(self.roots[i])
            return nodes
        return self.TraverseTree(self.root)
    
    def get_node_list(self):
        return self.nodeList
    
    def get_depth(self):
        if self.config["surrogate"]["multi_tree"]:
            return self.max_depths
        return self.max_depth
    
    def get_breadth(self):
        if self.config["surrogate"]["multi_tree"]:
            breadths = {}
            for i in self.config["picture"]["class_names"]:
                
                breadths[i] = self.get_breadth_multi(i)
            return breadths
        else:
            return self.get_breadth_single()
    def get_breadth_single(self):
        breadths = [0 for i in range(self.max_depth + 1)]
        for node in self.nodeList:
            breadths[node.depth] += 1
        return max(breadths)
    
    def get_breadth_multi(self,clas):
   
        breadths = [0 for i in range(self.max_depths[clas] + 1)]
        for node in self.nodeList[clas]:
            breadths[node.depth] += 1
        return max(breadths)
    
    def get_avg_representation(self):
        if self.config["surrogate"]["multi_tree"]:
            reps = {}
            for i in self.config["picture"]["class_names"]:
                reps[i] = self.get_avg_represenetation_multi(i)
            return reps
        else:
            return self.get_avg_representation_single()
    def get_avg_represenetation_multi(self, clas):
        Running_Avg = 0
        for node in self.nodeList[clas]:

            Running_Avg += node.represented_nodes
        return Running_Avg/len(self.nodeList[clas])
    def get_avg_representation_single(self):
        Running_Avg = 0
        
        for node in self.nodeList:
            Running_Avg += node.represented_nodes
        return Running_Avg/len(self.nodeList)
    
    def get_metrics(self):
        return {"Representation": self.get_avg_representation(), "Depth": self.get_depth(), "Breadth": self.get_breadth()}
    
    
    # def printer(self):
    #     if self.is_leaf:
    #         return str(self.return_value)
    #     else:
    #         obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
    #         return str(obj_to_Print) +  "\n" + "left: "+ self.left_node.printer() + "    right:"   + self.right_node.printer()
            
    # def self_print(self):
    #     obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
    #     print(obj_to_Print)
    
    # def get_parent(self):
    #     return self.parent_node