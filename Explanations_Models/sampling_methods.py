import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np
from abc import ABC, abstractmethod


class BaseSampler(ABC):
    def __init__(self):
        ABC.__init__(self)
        
    
    @abstractmethod
    def sample(self, N,point, variable_of_interest = "all"):
        """
        this takes in a point (typically in terms of state variables) and samples around it
        point: must be a numpy vector/ vector of vectors
        variable_of_interest: change to get information over the vector being samples
        """
        pass
    def get_samples(self, N, point, variable_of_interest = "all"):
        samples = self.sample(N, point, variable_of_interest)
        
        return samples
    

class GaussianSampler(BaseSampler):
    def __init__(self, variance):
        BaseSampler.__init__(self)
   
        self.variance = variance
    
    def sample(self, N, point, variable_of_interest):
        if variable_of_interest == "all":
            return np.random.normal(loc = point, scale = self.variance,  size=(N, point.shape[0]))
        else:
            sampled_points = np.zeros((N, point.shape[0]))
            for i in range(point.shape[0]):
                if i in variable_of_interest:
                    sampled_points[:, i] = np.random.normal(loc=point[i], scale=self.variance, size=N)
                else:
                    sampled_points[:, i] = point[i]
            return sampled_points

class UniformSampler(BaseSampler):
    def __init__(self):
        BaseSampler.__init__(self)
    def sample(self, N, point, variable_of_interest):
        if variable_of_interest == "all":
            return np.random.uniform(loc = point,  size=(N, point.shape[0]))
        else:
            sampled_points = np.zeros((N, point.shape[0]))
            for i in range(point.shape[0]):
                if i in variable_of_interest:
                    sampled_points[:, i] = np.random.uniform(loc=point[i], size=N)
                else:
                    sampled_points[:, i] = point[i]
            return sampled_points
