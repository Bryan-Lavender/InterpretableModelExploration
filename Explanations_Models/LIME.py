import torch
from Explanations_Models.sampling_methods import GaussianSampler, UniformSampler, UniformPolicySampler
from Explanations_Models.surrogate_models import LassoRegression
import numpy as np

class LimeModel():
    def __init__(self, model, point, config):
        self.config = config
        self.model = model.to(config["model_training"]["device"])
        self.point = point
        self.sampler = GaussianSampler(1)
        if config['sampling']['sampler'] == "Gaussian":
            self.sampler = GaussianSampler(config['sampling']['STD'])
        elif config['sampling']['sampler'] == "Uniform":
            self.sampler = UniformSampler()
        elif config['sampling']['sampler'] == "UniformPolicy":
            self.sampler = UniformPolicySampler()
        else:
            print("bad sampler")
            exit(1)
        self.interpretable_models = []

        if config['surrigate_params']['model_type'] == "Lasso":
            for i in range(0, config['sampling']['output_num']):
                self.interpretable_models.append(LassoRegression(self.point,config['surrigate_params']).to(config["model_training"]["device"]))

        self.sample_points = None
    
    def sample(self, config):
        self.sample_points = torch.tensor(self.sampler.sample(config), device=self.config["model_training"]["device"], dtype=torch.float32)

    
    def fit_surrigate(self, X, Y):
        out = 0
        for i in self.interpretable_models:
            i.fit(X, Y[:, out])
            out += 1
    
    def runner(self):
        self.sample(self.config['sampling'])
        with torch.no_grad():
            Y = self.model(self.sample_points)
        self.fit_surrigate(self.sample_points, Y)
        
        ct = 0
        acc_arr = []
        for i in self.interpretable_models:
            torch.save(i.linear.state_dict(), self.config["explanation_weights"]["model_path"] + self.config["explanation_weights"]["outputs"][ct] + ".pt")
            acc_arr.append(i.evaluate(self.sample_points, Y[:, ct]))
            ct += 1
        np.save(self.config["explanation_output"]["MAE_MSE_RMSE_Rsq"],np.array(acc_arr))

    
        
        
