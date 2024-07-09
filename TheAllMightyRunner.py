import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient
from EnvRunner import GymRunner
import random
import yaml
yaml.add_constructor("!join", join)
parser = argparse.ArgumentParser()

parser.add_argument("--config_filename", required=False, type=str)
parser.add_argument("--plot_config_filename", required=False, type=str)
parser.add_argument("--run_basic_tests", required=False, type=bool)

def weight_dict_corrector(loaded_state_dict):
    new_state_dict = {}
    for key, value in loaded_state_dict.items():
        new_key = key.replace("network.", "")  # Remove the 'network.' prefix
        new_state_dict[new_key] = value
    return new_state_dict

if __name__ == "__main__":

    args = parser.parse_args()
    config_file = open("config_envs/{}.yml".format(args.config_filename))
    config = yaml.load(config_file, Loader=yaml.FullLoader)

    env = gym.make(config["env"]["env_name"], render_mode="rgb_array")
    seed = config["env"]["seed"][0]
    
    runner_thang = GymRunner(config, env)
    model = PolicyGradient(runner_thang.runner, runner_thang.recorder, config, seed)
    runner_thang.init_model(model)

    print(model.sample_path(2))

