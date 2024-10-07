import torch
from torch import nn
import numpy as np
from .methods.LRP import TwoDWeights_LRPModel
methods = {
    "LRP": TwoDWeights_LRPModel
}
class FeatureImportance():
    def __init__(self, config, model):
        self.config = config
        self.model = model

    
    def Calc_Relevence_Importance(self, input_point, printer = False):
        if self.config["FI"]["method"] == "LRP":
            Model = methods[self.config["FI"]["method"]](self.model)
            out, FI_matrix = Model.get_FI(input_point, printer)
            FI_matrix = torch.stack(FI_matrix).detach().numpy()
            out = out.detach().numpy()
            return out, FI_matrix
    
    def Relevence(self, X):
        X = torch.tensor(X)
        FI = []
        out = []
        for i in X:
            out_t, tmpFI = self.Calc_Relevence_Importance(i)
            out.append(out_t)
            tmpFI = np.abs(tmpFI)
            for j in range(tmpFI.shape[0]):
                tmpFI[j,:]= (tmpFI[j,:] - np.min(tmpFI[j,:])) / (np.max(tmpFI[j,:]) - np.min(tmpFI[j,:]))
            FI.append(tmpFI)
        FI = np.stack(FI)
        out = np.stack(out)
        return out, FI
    


        

