import shap
import numpy as np
import torch
class SHAP():
    def __init__(self, model):
        self.model = model
        self.explainer = None
    def _f(self, X):
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32)
            return self.model.forward(X).cpu().numpy()

    def get_single_FI(self, point):
        if self.explainer == None:
            tmp_explainer = shap.KernelExplainer(self._f, point)
            shap_values = tmp_explainer.shap_values(point, nsamples=1000)
            shap_values = np.array(shap_values).squeeze()
            return self.model.forward(point),shap_values
        else:
            shap_values = self.explainer.shap_values(point, nsamples=1000)
            shap_values = np.array(shap_values).squeeze()
            return self.model.forward(torch.tensor(point, dtype=torch.float32)),shap_values
            
    
    def fit_kernel_exp(self, X):
        self.explainer = shap.KernelExplainer(self._f, X)

    def get_fi(self, X, use_kernel = True):
        
        if self.explainer == None and use_kernel:
            self.fit_kernel_exp(X)
        elif self.explainer != None and not use_kernel:
            self.explainer = None
        
        out = []
        FIs = []

        for i in X:
            y, FI = self.get_single_FI(i)
            out.append(y)
            FIs.append(FI)

        return out, np.stack(FIs)
        
            
    
    
