import torch
from Explanations_Models.sampling_methods import GaussianSampler, UniformSampler
from Explanations_Models.surrogate_models import LassoRegression
class LimeModel():
    def __init__(self, model, point, config):
        self.config = config
        self.model = model
        self.point = point
        self.sampler = GaussianSampler(1)
        if config['sampling']['sampler'] == "Gaussian":
            self.sampler = GaussianSampler(config['sampling']['STD'])
        elif config['sampling']['sampler'] == "Uniform":
            self.sampler = UniformSampler()
        self.interpretable_models = []

        if config['surrigate_params']['model_type'] == "Lasso":
            for i in range(0, config['sampling']['output_num']):
                self.interpretable_models.append(LassoRegression(self.point,config['surrigate_params']))

        self.sample_points = None
    
    def sample(self, n, x, variables = "all"):
        self.sample_points = torch.tensor(self.sampler.sample( n, x, variables), device="cuda", dtype=torch.float32)
    
    
    def fit_surrigate(self, X, Y):
        out = 0
        for i in self.interpretable_models:
            i.fit(X, Y[:, out])
            out += 1
    
    def runner(self):
        self.sample(self.config['sampling']['sample_number'], self.point, self.config['sampling']['variables'])
        with torch.no_grad():
            Y = self.model(self.sample_points)
        self.fit_surrigate(self.sample_points, Y)
        
