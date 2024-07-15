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
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config["env"]["env_name"])
        seed = config["env"]["seed"]
        self.model = PolicyGradient(self.runner, self.recorder, config, seed)
        self.video_tag = 0


    # def __init__(self, config, env, model = None):
    #     self.env = env
    #     self.model = model
    #     self.config = config
    #     self.video_tag = 0
    """
    init runner:
    args: 
        env (gym environment) (optional)
    returns:
        dictionary with observations, actions and rewards from a single episode

    """
    def load_weights(self, PATH = None):
        if PATH == None:
            PATH_actor = self.config["output"]["actor_output"].format(self.config["env"]["seed"])
            PATH_critic = self.config["output"]["critic_output"].format(self.config["env"]["seed"])
        if torch.cuda.is_available():
            self.model.baseline_network.load_state_dict(torch.load(PATH_critic))
            self.model.policy.load_state_dict((torch.load(PATH_actor)))
        else:
            self.model.baseline_network.load_state_dict(torch.load(PATH_critic,map_location=torch.device('cpu')))
            self.model.policy.load_state_dict((torch.load(PATH_actor,map_location=torch.device('cpu'))))
        
    def init_model(self, model):
        self.model = model

    def actNoDist(self,state):
        with torch.no_grad():
            state = torch.tensor(state, dtype = torch.float32).to(self.model.device)
            return torch.argmax(self.model.network(state)).cpu().numpy()

    def runner(self, env = None, use_dist = True, model = None, seed = None):
        states = []
        actions = []
        rewards = []
        if env == None:
            env = self.env
        else:
            env = env

        if model == None and use_dist:
            act = self.model.policy.act
        elif model == None and not use_dist:
            act = self.actNoDist
        elif model!= None:
            act = model.forward

        if seed!= None:
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()
        for i in range(self.config["hyper_params"]["max_ep_len"]):
            action = act(state)
            states.append(state)
            actions.append(action)
            state, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated:
                break
            
        return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions)}

    def recorder(self, env = None, use_dist = True, model = None, filename = None):
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
        if filename == None:
            env = gym.wrappers.RecordVideo(
                env,
                self.config["output"]["record_path"].format(self.config["env"]["seed"]) + "_" + str(self.video_tag),
                step_trigger=lambda x: x == 0,
            )
        else:
            env = gym.wrappers.RecordVideo(
                env,
                filename,
                step_trigger=lambda x: x == 0,
            )
        self.video_tag += 1
        self.runner(env = env, model = model)
        env.close()
    
    def comparitor(self, use_dist = False, model1 = None, model2 = None):

        env1 = gym.make(self.config["env"]["env_name"])
        env2 = gym.make(self.config["env"]["env_name"])
        seed = np.random.randint(0, 1000)
        
        if model1 == None:
            model1 = None
        if model2 == None:
            model2 == self.model
        path1 = self.runner(env1, use_dist=use_dist, model=model1, seed = seed)
        path2 = self.runner(env2, use_dist=use_dist, model=model2, seed = seed)

        mae = np.mean(np.abs(path1["observation"] - path2["observation"]))
        mse = np.mean((path1["observation"] - path2["observation"])**2)
        print("MSE: ", mse)
        print("MAE: ", mae)

    



