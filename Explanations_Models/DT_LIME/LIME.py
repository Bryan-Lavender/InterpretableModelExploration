from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from .sampling_methods import Policy_Sampler, Uniform_Sampler, Gaussian_Sampler
import torch
import numpy as np
from sklearn.tree import plot_tree
import matplotlib
import matplotlib.pyplot as plt
class DecisionTree():
    def __init__(self, config, env_runner):
        if config["surrogate"]["classifier"]:
            self.model = DecisionTreeClassifier(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["min_split"])
        else:
            self.model = DecisionTreeRegressor(criterion=config["surrogate"]["criterion"], min_samples_split=config["surrogate"]["min_split"])
        
        self.env_runner = env_runner
        self.config = config
    
    def fit(self,X, Y):
        self.model.fit(X, Y)
    
    def forward(self, X):
        is_pass = False
        if type(X) == torch.Tensor:
            X = X.to("cpu").numpy()
        if len(X.shape) == 1:
            is_pass = True
            X = np.array([X])
            
            if self.config["surrogate"]["classifier"]:
                return self.model.predict(X)[0]
            else:
                return np.argmax(self.model.predict(X)[0])
        return self.model.predict(X)

    def display_tree(self):
        fig = plt.figure(figsize=((25,20)))
        plot_tree(self.model, 
                feature_names = ["x", "vel", "angle", "angle_vel"],
                class_names = ["left", "right"],
                impurity=False,
                proportion=True,
                filled=True
                )
        fig.savefig("daTreeMan.png")

samplers = {
    "Policy"  : Policy_Sampler,
    "Uniform" : Uniform_Sampler,
    "Gaussian": Gaussian_Sampler,
}

class LIME():
    def __init__(self, config, env_runner):
        self.config = config
        self.comparitor = env_runner.comparitor
        self.env_runner = env_runner.runner
        self.surr_model = DecisionTree(self.config, self.env_runner)
        if config["sampler"]["use_dist"]:
            self.deep_model = env_runner.model.policy
        else:
            self.deep_model = env_runner.model.network
        self.device = env_runner.model.device
        self.sampler = samplers[config["sampler"]["sample_type"]](config, self.env_runner)
        
    
    def train(self):
        with torch.no_grad():
            samples = self.sampler.sample().to(self.device)
            if self.config["surrogate"]["classifier"]:
                Y = torch.argmax(self.deep_model.forward(samples), dim = 1).to("cpu").numpy()
            else:
                Y = self.deep_model.forward(samples).to("cpu").numpy()
            samples = samples.to("cpu").numpy()
            X = samples
            print("Dist: 0's / 1's", len(Y[Y==0]), len(Y[Y==1]))
            self.surr_model.fit(X,Y)
    
    #can only test action and state metrics
    def percent_Correct(self, print_val = True):
        path = self.env_runner(use_dist = self.config["sampler"]["use_dist"])
        y_pred = self.surr_model.forward(path["observation"])
        y_true = path["action"]
        TP = 0
        if self.config["surrogate"]["classifier"]:
            for i in range(self.config["env"]["action_dim"]):
                TP += np.sum((y_true == i) & (y_pred == i))
            

        # Calculate percentages
        total = len(y_true)
        percent_correct = TP/total
        if(print_val):
            print(f"Percent Correct: {percent_correct:.2f}%")
        return percent_correct
    
    def absolute_distance(self, print_val= True):
        self.comparitor(use_dist=self.config["sampler"]["use_dist"], model2 = self.surr_model)

        
    
        

    
    
        

    