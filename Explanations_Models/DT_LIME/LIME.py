from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .sampling_methods import Policy_Sampler, Uniform_Sampler, Gaussian_Sampler
import torch

class DecisionTree():
    def __init__(self, config, env_runner):
        if config["surrogate"]["classifier"]:
            self.model = DecisionTreeClassifier(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["criterion"])
        else:
            self.model = DecisionTreeRegressor(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["criterion"])
        
        self.env_runner = env_runner
        self.config = config
    
    def fit(self,X, Y):
        self.model.fit(X, Y)
    
    def forward(self, X):
        if type(X) == torch.Tensor:
            X = X.to("cpu").numpy()
        
        return self.surr_model.predict(X)

samplers = {
    "Policy"  : Policy_Sampler,
    "Uniform" : Uniform_Sampler,
    "Gaussian": Gaussian_Sampler,
}

class LIME():
    def __init__(self, config, env_runner):
        self.config = config
        self.env_runner = env_runner.runner
        self.surr_model = DecisionTree(self.config, self.env_runner)
        self.deep_model = env_runner.model.network
        if config["sampler"]["sample_type"] == "Policy":
            self.sampler = samplers[config["sampler"]["sample_type"]](config, self.env_runner)
        else:
            self.sampler = samplers[config["sampler"]["sample_type"]](config)
    
    def train(self):
        samples = self.sampler.sample().to(self.config["model_training"]["device"])
        Y = self.deep_model.forward(samples).to("cpu").numpy()
        samples = samples.to("cpu").numpy()
        X = samples

        self.surr_model.fit(X,Y)
    
    #can only test action and state metrics
    def test(self):
        path = self.env_runner()
    
    
        

    
    
        

    