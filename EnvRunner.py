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

import random
import yaml
"""
Begin Environment runner.
Creates a class to exectute an environment following a model's policy.
General Runner: needs a config file, environment, and model

MODEL must have: model.policy.act(state)


different possible env's: 
GymRunner: for openAI's Gym library
"""
class GymRunner:
    def __init__(self, config, env, model = None):
        self.env = env
        self.model = model
        self.config = config
    """
    init runner:
    args: 
        env (gym environment) (optional)
    returns:
        dictionary with observations, actions and rewards from a single episode

    """
    def init_model(self, model):
        self.model = model
    
    def runner(self, env = None):
        states = []
        actions = []
        rewards = []
        if env == None:
            env = self.env
        else:
            env = env
        state, info = env.reset()
        for i in range(self.config["hyper_params"]["max_ep_len"]):
            action = self.model.policy.act(state)
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated:
                break
            
        return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions)}

    def recorder(self):
        """
        Recorder:
        Creates video of an execution. Uses a temporary env from GYM to do so.
        Saves it in self.config["output"]["record_path"]
        """
        env = gym.make(
            self.config["env"]["env_name"],
            render_mode="rgb_array"
        )
        env.reset(seed=self.config["env"]["seed"])
        env = gym.wrappers.RecordVideo(
            env,
            self.config["output"]["record_path"].format(self.config["env"]["seed"]),
            step_trigger=lambda x: x % 100 == 0,
        )
        self.runner(env = env)

    




            