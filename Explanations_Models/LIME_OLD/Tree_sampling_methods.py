import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from array import array

class Gaussian_Sampler():
    def __init__(self, config):
        self.config = config

    def sample(self, mean = 0):
        num_samples = self.config["sampler"]["num_samples"]
        if type(mean) == int:
            return torch.normal(mean = mean, std = self.config["sampler"]["std"], size = ( num_samples,self.config["env"]["obs_dim"]))
        
        else:
            means_repeated = mean.repeat(1, num_samples, 1)
            samples = torch.normal(means_repeated, self.config["sampler"]["std"])
            return samples

class Uniform_Sampler():
    def __init__(self, config):
        self.config = config
    def sample(self):
        num_samples = self.config["sampler"]["num_samples"]
        bounds = torch.tensor(self.config["sampler"]["bounds"], dtype = torch.float32)
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        uniform_samples = torch.rand((num_samples, bounds.shape[0]))
        scaled_samples = lower_bounds + uniform_samples * (upper_bounds - lower_bounds)
        return scaled_samples
    
class Policy_Sampler():
    def __init__(self, config, runner):
        self.config = config
        self.runner = runner
    def sample(self):
        num_samples = self.config["sampler"]["num_samples"]
        path = self.runner()
        return torch.tensor(path["observation"])
