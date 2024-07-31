import argparse
import os
import sys
import numpy as np
import torch
import gym
import matplotlib
import importlib
import supersuit

matplotlib.use("agg")
import matplotlib.pyplot as plt
import unittest
from DeepLearning_Models.utils.general import join, plot_combined
from DeepLearning_Models.ActorCritic.policy_gradient import PolicyGradient

# test
from wrappers import custom_video_wrapper

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
class PettingZooRunner:
    def __init__(self, config, env, module, model = None):
        self.env = env
        self.model = model
        self.config = config
        self.module = module
        self.video_tag = 0
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

        env.reset()
        for i in range(self.config["hyper_params"]["max_ep_len"]):

            for agent in env.agent_iter():

                state, reward, terminated, truncated, info = env.last()

                if terminated:
                    action = None
                
                else:
                    if self.config["network"]["network_type"] == "cnn":
                        state = state.astype('float32')
                        state = np.array([np.transpose(state, (2, 0, 1))])
                        action = self.model.policy.act(state)[0]
                        state = state[0]

                    else:
                        action = self.model.policy.act(state)

                    action = np.clip(action, -1.0, 1.0)
                    
                states.append(state)

                if (action is None):
                    actions.append(np.array([0,0,0,0]))
                
                else:
                    actions.append(action)

                env.step(action)
                rewards.append(reward)

                if terminated:
                    break
            
        return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions)}

    def recorder(self, i = 0):
        """
        Recorder:
        Creates video of an execution. Uses a temporary env from GYM to do so.
        Saves it in self.config["output"]["record_path"]
        """

        env = getattr(
            self.module, 
            self.config["env"]["env_name"]
        ).env(render_mode='rgb_array')

        env.reset(seed=self.config["env"]["seed"])

        env = custom_video_wrapper.RecordVideo(
            env,
            self.config["output"]["record_path"].format(self.config["env"]["seed"]) + "_" + str(self.video_tag),
            step_trigger=lambda x: x == 0,
        )

        self.video_tag += 1
        self.runner(env = env)
        env.close()

            