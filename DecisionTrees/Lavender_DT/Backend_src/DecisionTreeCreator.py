import numpy as np
from .Criterion import entropy, MSE
from .ShouldCreateLeaf import one_class_left, STD_diff
from .SplittingFunctions import regular_info_gain, ImportanceWeighing, ImportanceMinimization
from .WeighingFunctions import Variance_Weighted_sum, Max_Average_FI, Max_Over_Outcome, Double_Average, Classification

criterions = {"entropy": entropy, "MSE": MSE}
leaf_creator = {"single_class": one_class_left, "STD": STD_diff}
splitting_functions = {"normal": regular_info_gain, "ImportanceMinimization": ImportanceMinimization, "ImportanceWeighing": ImportanceWeighing}
weighing_methods = {"normal": None, "Var_Weighted": Variance_Weighted_sum, "Max_Avg": Max_Average_FI, "Max_All": Max_Over_Outcome, "Double_Avg":Double_Average, "Class": Classification}
class DecisionTreeCreator():
    def __init__(self, config):
        self.criterion = criterions[config["criterion"]]
        self.leaf_creator = leaf_creator[config["leaf_creator"]]
        self.splitting_function = splitting_functions[config["splitting_function"]]
        
        self.minimization = False
        if config["splitting_function"] == "ImportanceMinimization":
            self.minimization = True
        self.use_fi = True

        if config["splitting_function"] == "normal":
            self.use_fi = False

        self.weighing_method = weighing_methods[config["weighing_method"]]

        self.object_names = config["object_names"]
        self.tree = {}

    def fit(self, X, Y, FI, out_logits):
        n_samples, n_features = X.shape
        self.X = X
        self.Y = Y

        root_sorted_indices = {j: np.argsort(X[:, j]) for j in range(n_features)}

        stack = [(self.tree, None, np.arange(n_samples), root_sorted_indices, 0)]

        while stack:
            parent, split_dir, indices, sorted_indices, depth = stack.pop()

            X_node = X[indices]
            Y_node = Y[indices]
            
            
            leaf_val = self.leaf_creator(Y_node)
            if leaf_val is not None:
                parent[split_dir] = {"value": leaf_val, "depth": depth, "represented": len(Y_node)}
                continue
            
            
            
            if FI is not None:
                FI_node = FI[indices]
                out_node = out_logits[indices]

            weights = None
            weight = None
            if not self.minimization and self.use_fi:
                weights = self.weighing_method(FI_node, out_node)
                
                

            best_gain = -np.inf if not self.minimization else np.inf
            best_feature = None
            best_threshold = None
            best_left = None
            best_right = None
        
            for j in range(n_features):
                sorted_idx = sorted_indices[j]
                sorted_idx = sorted_idx[np.isin(sorted_idx, indices, assume_unique=True)]
                x_sorted = X[sorted_idx, j]
                y_sorted = Y[sorted_idx]
                if self.use_fi:
                    FI_sorted = FI[sorted_idx]
                    out_logit_sorted = out_logits[sorted_idx]

                for i in range(1, len(sorted_idx)):
                    if x_sorted[i] == x_sorted[i - 1]:
                        continue

                    threshold = (x_sorted[i] + x_sorted[i - 1]) / 2
                    y_left = y_sorted[:i]
                    y_right = y_sorted[i:]
                    
                    if self.minimization:
                        weight_L = self.weighing_method(FI_sorted[:i], out_logit_sorted[:i])
                        weight_R = self.weighing_method(FI_sorted[i:], out_logit_sorted[i:])
                        weight = (weight_L[j], weight_R[j])
                    elif self.use_fi:
                        weight = weights[j]
                    else:
                        weight = 1
                    gain = self.splitting_function(y_sorted, y_left, y_right, weight, self.criterion)
                    if (self.minimization and gain < best_gain) or (not self.minimization and gain > best_gain):
                        best_gain = gain
                        best_feature = j
                        best_threshold = threshold
                        best_left = sorted_idx[:i]
                        best_right = sorted_idx[i:]

            if self.object_names == None:
                node = {'feature':best_feature, 'threshold':best_threshold}
            else:
                node = {'feature':best_feature, "feature_name": self.object_names[best_feature], 'threshold':best_threshold}

            if parent == {}:
                self.tree = node
            else:
                parent[split_dir] = node

            left_sorted = {
                j: sorted_indices[j][np.isin(sorted_indices[j], best_left, assume_unique=True)]
                for j in range(n_features)
            }
            right_sorted = {
                j: sorted_indices[j][np.isin(sorted_indices[j], best_right, assume_unique=True)]
                for j in range(n_features)
            }
          
            stack.append((node, 'right', best_right, right_sorted, depth + 1))
            stack.append((node, 'left', best_left, left_sorted, depth + 1))

    