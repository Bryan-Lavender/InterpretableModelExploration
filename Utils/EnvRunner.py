
import numpy as np
import torch
import gym

"""
Begin Environment runner.
Creates a class to exectute an environment following a model's policy.
General Runner: needs a config file, environment, and model

MODEL must have: model.policy.act(state)


different possible env's: 
GymRunner: for openAI's Gym library
"""
class GymRunner:
    def __init__(self, config, model, seed = None):
        self.config = config
        self.env = gym.make(config["env"]["env_name"])
        self.seed = seed
        self.model = model
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

    def init_model(self, model):
        self.model = model

    def actNoDist(self,state, get_activations = False):
        with torch.no_grad():
            if get_activations:
                # if self.config["env"]["env_name"] == "LunarLander-v2":
                #     state = state[0]
                state = torch.tensor(state, dtype = torch.float32).to(self.model.device)
                out = self.model.network(state)
            
                return out.cpu().numpy(), torch.argmax(out).cpu().numpy()
            else:
                state = torch.tensor(state, dtype = torch.float32).to(self.model.device)
                out = self.model.network(state)
                if self.config["env"]["discrete"]:
                    return torch.argmax(out).cpu().numpy()
                return out.cpu().numpy()

    #envRunner made specifically for gym 0.24 because they are horrible at setting standards
    def runner(self, env=None, use_act = False, model = None, seed = None, return_activations = False):
        if env == None:
            env = self.env
        
        if model == None and use_act:
            act = self.model.policy.act
        elif model == None:
            act = self.actNoDist
        elif model is not None:
            act = model.act
        
        if seed is not None:
        
            state, info = env.reset(seed=seed)
        else:
            state, info = env.reset()
        
        states = []
        actions = []
        rewards = []
        activations = []

        for i in range(self.config["hyper_params"]["max_ep_len"]):
            if return_activations:
                activation, action = act(state, get_activations = True)
                activations.append(activation)
            else:
                action = act(state)
            states.append(state)
            actions.append(action)
            
            step_result = env.step(action)

            if len(step_result) == 5:
                state, reward, terminated, truncated, info = step_result
                done = terminated
            else:
                state, reward, done, info = step_result

            rewards.append(reward)

            if done:
                break
        
        if return_activations:
            return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions), "activations": np.stack(activations)}
            
        return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions)}


                
    # def runner(self, env = None, use_dist = False, model = None, seed = None):
    #     weird_box2d_state = True
    #     # if "Acrobot" in self.config["env"]["env_name"] or "Bipedal" in self.config["env"]["env_name"] or "Lunar" in self.config["env"]["env_name"]:
    #     #     weird_box2d_state = True
    #     if self.config["env"]["discrete"] and use_dist == False:
    #         get_output = True
    #     else:
    #         get_output = False

    #     states = []
    #     actions = []
    #     rewards = []
    #     activations = []

    #     if env == None:
    #         env = self.env
    #     else:
    #         env = env
        
    #     if model == None and use_dist:
    #         act = self.model.policy.act
    #     elif model == None and not use_dist:          
    #         act = self.actNoDist
    #     elif model!= None:
    #         act = model.act
    #     if weird_box2d_state:
    #         if seed!= None:
    #             env.seed(seed)
    #             state = env.reset()
    #         else:
    #             state = env.reset()
    #     else:
    #         if seed!= None:
    #             state, info = env.reset(seed=seed)
    #         else:
    #             state, info = env.reset()
    #     print(state)
    #     for i in range(self.config["hyper_params"]["max_ep_len"]):
    #         if get_output:
    #             output, action = act(state)
    #         else:
    #             action = act(state)
                
    #         states.append(state)
    #         actions.append(action)
    #         if get_output:
    #             activations.append(output)
    #         if weird_box2d_state:
    #             state, reward, terminated, truncated = env.step(action)
    #         else:
    #             state, reward, terminated, truncated, info = env.step(action)
    #         rewards.append(reward)

    #         if terminated:
    #             break
    #     if get_output:

    #         TR = {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions), "activations": np.stack(activations)}
    #         return TR
    #     return {"observation": np.array(states), "reward": np.array(rewards), "action": np.array(actions)}

    def recorder(self, env = None, use_dist = True, model = None, filename = None, seed = None):
        """
        Recorder:
        Creates video of an execution. Uses a temporary env from GYM to do so.
        Saves it in self.config["output"path"]
        """
        env = gym.make(
            self.config["env"]["env_name"],
            render_mode="rgb_array"
        )
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
        self.runner(env = env, model = model, seed=seed)
        env.close()
