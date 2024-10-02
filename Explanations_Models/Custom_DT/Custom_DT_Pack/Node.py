import numpy as np
from .BucketAlgs import half_split
from .SplittingFunctions import entropy, MSE, MAE
from scipy.spatial.distance import pdist
from .FI_Calulator import Policy_Smoothing_Max_Weighting, Policy_Smoothing_MaxAvg_Weighting, Policy_Smoothing_Var_Weighting, Classification_Forgiveness
bucketing_mech = {"half_split": half_split}
splitting_functions = {"entropy": entropy, "MSE": MSE, "MAE": MAE}
FICalcs = {"Var_Weighted": Policy_Smoothing_Var_Weighting, "Max_all": Policy_Smoothing_Max_Weighting, "max_avg":Policy_Smoothing_MaxAvg_Weighting, "class": Classification_Forgiveness}
class Single_Attribute_Node():
    #begin Single_Attribute_Node: X (input), Y (output), feature_list ([disc, cont, disc, disc, cont...])
    """
    Global_Vars: 
        feature_index: index of split
        is_leaf: leaf node to return
        vals: values of bound on feature OR values to return
    """
    def __init__(self, config, parent_node = None):
        self.feature_index = None
        self.is_leaf = False
        self.val_bucket = None
        self.return_value = None
        self.config = config
        self.parent_node = parent_node
    
    def fit(self, X,Y, depth=None, FI = None, out_logits = None):
        
        if depth != None:
            self.depth = depth + 1
        
        if self.config["surrogate"]["classifier"] and len(Y[Y.keys()[0]].unique()) == 1:
            print(len(Y))
            self.return_value = Y[Y.keys()[0]].value_counts().index[0]
            self.is_leaf = True

            self.left_node = -1
            self.right_node = -1
            return {"Value": self.return_value},self.depth
        
        elif not self.config["surrogate"]["classifier"] and (len(Y)==1 or max(pdist(Y.values, metric="euclidean")) <= self.config["surrogate"]["tree_cutoff"] ):
            if len(Y) == 1:
                self.return_value = Y.to_numpy()[0]
            else:
                print(len(Y))
                self.return_value = np.mean(Y.to_numpy(), axis=0)
                

            self.is_leaf = True

            self.left_node = -1
            self.right_node = -1
            return {"Value": self.return_value},self.depth
        else:
            buckets = self.bucket(X)
            
            min_var, min_bucket, min_val,indicies_left, indicies_right = self.split(X,Y, buckets, FI=FI, out_logits=out_logits)
            self.feature_index = min_var
            self.val_bucket = min_bucket
            self.heuristic_value = min_val
            
     
            self.left_node = Single_Attribute_Node( self.config, parent_node=self)
            self.right_node = Single_Attribute_Node( self.config, parent_node=self)
           
            if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                left_dict, left_max = self.left_node.fit(X.loc[indicies_left], Y.loc[indicies_left], self.depth, FI=FI.loc[indicies_left], out_logits=out_logits.loc[indicies_left])
                right_dict, right_max = self.right_node.fit(X.loc[indicies_right], Y.loc[indicies_right],self.depth, FI=FI.loc[indicies_right], out_logits=out_logits.loc[indicies_right])
            else:
                left_dict, left_max = self.left_node.fit(X.loc[indicies_left], Y.loc[indicies_left], self.depth)
                right_dict, right_max = self.right_node.fit(X.loc[indicies_right], Y.loc[indicies_right],self.depth)
         
            return {"Feature": self.feature_index, "Bucket": self.val_bucket, "Left_Child": left_dict, "Right_Child": right_dict}, max(left_max, right_max)
    
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
    
    def split(self, X, Y, buckets, FI = None, out_logits = None):
        #heuristics = {}
        min_var = None
        min_bucket = None
        min_val = None
        
        if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
            FI_val = FICalcs["Var_Weighted"](FI, out_logits)
        for var in buckets.keys():
                
            #heuristics[var] = []
            #if len(buckets[var]) != 1:
                for bucket in buckets[var]:
                    # calculate entropy 
                    
                    if type(bucket) == np.float64 or type(bucket) == np.float32 or type(bucket) == float:
                        indicies_left = X[X[var] <= bucket].index
                    else:
                        indicies_left = X[X[var] == bucket].inde
                    
                    indicies_right = X.index.difference(indicies_left)

                    
                    
                    #indicies_left = np.array(indicies_left)
                    #indicies_right = np.array(X.index.difference(indicies_left))
                    
                    Y_left = Y.loc[indicies_left]
                    Y_right = Y.loc[indicies_right]
                
                
                    heuristic_left  = 0 if len(indicies_left) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_left)
                    heuristic_right = 0 if len(indicies_right) == 0 else splitting_functions[self.config["surrogate"]["criterion"]](Y_right)
                
                    
                    val = len(Y_left)/len(Y)*heuristic_left + len(Y_right)/len(Y)*heuristic_right
                    #heuristics[var].append(val)
                    
                    if self.config["surrogate"]["use_FI"] and type(FI) != type(None):
                        # FI_val = FICalcs["Var"](FI, out_logits)
                        val = val * (1/FI_val[var].iloc[0])
                        
                    if min_val == None or min_val > val:
                        min_val = val
                        min_var = var
                        min_bucket = bucket

                        curr_ind_left = indicies_left
                        curr_ind_right = indicies_right
            #else:

        return (min_var, min_bucket, min_val, curr_ind_left, curr_ind_right)

    
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
    
    def printer(self):
        if self.is_leaf:
            return str(self.return_value)
        else:
            obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
            return str(obj_to_Print) +  "\n" + "left: "+ self.left_node.printer() + "    right:"   + self.right_node.printer()
            
    def self_print(self):
        obj_to_Print = {"Feature": self.feature_index, "Value": self.val_bucket}
        print(obj_to_Print)
    
    def get_parent(self):
        return self.parent_node
    










    
