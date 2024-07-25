import yaml
import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
from EnvRunner import GymRunner
from Explanations_Models.DT_LIME.LIME import LIME
import json
import warnings

# Suppress all deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--task", required=True, type=str)

class MetricGetter():
    def __init__(self, config, Runner):
        self.config = config
        self.Runner = Runner
    
    def run_series(self):
        config = self.config
        PercentCorrect = []
        ExecutionMSE = []
        ExecutionDiff = []
        TDepth = []
        TBreadth = []
        top_splits = []
        for i in range(config["metric_hyperparameters"]["tree_execution_samples"]):
            limemod = LIME(config, Runner)
            limemod.train()
            PercentCorrect.append(limemod.percent_Correct())
            ADmse, CountDiff = limemod.absolute_distance()
            ExecutionMSE.append(ADmse)
            ExecutionDiff.append(CountDiff)

            depth, breadth = limemod.surr_model.depth_breadth()
            TDepth.append(depth)
            TBreadth.append(breadth)
            top_splits.append(config["picture"]["labels"][limemod.surr_model.get_top_split()])
        return {"PercentCorrect": PercentCorrect, "EpisodeDistance": ExecutionMSE, "Episode_Length_Distance": ExecutionDiff, "Depth": TDepth, "Breadth": TBreadth, "TopSplits": top_splits}

    def sample_rate(self):
        seq = self.config["metric_hyperparameters"]["sample_sequence"]
        returner = {}
        for i in seq:
            config["sampler"]["num_samples"] = i
            out = self.run_series()
            returner[str(i)] = out
        
        return returner

    def Saver(self, metrics):
        path = self.config["exp_output"]["output_path"]
        path = path + "/" + config["surrogate"]["criterion"] + ".json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as json_file:
            json.dump(metrics, json_file)
    

if __name__ == "__main__":
    args = parser.parse_args()
    config_file = open("config_envs/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    config.update(yaml.load(open("config_explanations/{}.yml".format(args.config_filename)), Loader= yaml.FullLoader))
    Runner = GymRunner(config)
    Runner.load_weights(PATH = None)
