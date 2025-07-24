import torch
from torch import nn
import numpy as np
from .methods.LRP import TwoDWeights_LRPModel
from .methods.FiniteDifferences import FiniteDifferences
methods = {
    "LRP": TwoDWeights_LRPModel,
    "FD": FiniteDifferences
}
class FeatureImportance():
    def __init__(self, method, model):
        self.method_name = method
        self.method = methods[method]
        self.model = model

    
    def Calc_Relevence_Importance(self, input_point, h = .5, printer = False):
        if self.method_name == "LRP":
            Model = self.method(self.model)
            out, FI_matrix = Model.get_FI(input_point, printer)
            FI_matrix = torch.stack(FI_matrix).detach().numpy()
            out = out.detach().numpy()
            return out, FI_matrix
        else:
            Model = self.method(self.model)
            out, FI_matrix = Model.Relevence(input_point, h = h)
            return out, FI_matrix
        
    def Relevence(self, X, h=10e-5):
        #print(h)
        X = torch.tensor(X, dtype=torch.float32)
        FI = []
        out = []
        for i in X:
            out_t, tmpFI = self.Calc_Relevence_Importance(i, h=h)
            out.append(out_t)
            tmpFI = np.abs(tmpFI)
            
            # for j in range(tmpFI.shape[0]):
            #     tmpFI[j,:]= (tmpFI[j,:] - np.min(tmpFI[j,:])) / (np.max(tmpFI[j,:]) - np.min(tmpFI[j,:]))
            FI.append(tmpFI)
        FI = np.concatenate(FI)
        FI = (FI - np.min(FI)) / (np.max(FI) - np.min(FI))
        FI = np.split(FI, len(X))
        FI = np.stack(FI)
        out = np.stack(out)
        return out, FI




        

