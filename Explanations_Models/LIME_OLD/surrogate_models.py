import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import joblib

class DecisionTree():
    def __init__(self, config, env_runner):
        if config["surrogate"]["classifier"]:
            self.surr_model = DecisionTreeClassifier(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["criterion"])
        else:
            self.surr_model = DecisionTreeRegressor(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["criterion"])
        
        self.env_runner = env_runner
        self.config = config
    
    def fit(self,X, Y):
        self.surr_model.fit(X, Y)
    
    def forward(self, X):
        if type(X) == torch.Tensor:
            X = X.to("cpu").numpy()
        
        return self.surr_model.predict(X)
        
        
