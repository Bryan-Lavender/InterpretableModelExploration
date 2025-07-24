import numpy as np
import torch
import pandas as pd
from .Sampler import trajectory_sampler, uniform_sample_policy

from .EnvRunner import GymRunner
import random

def Trajectory(config, DTModel, MainModel, trajectory_number = 1, loaded_set = None):
    if config["surrogate"]["classifier"]:
        return TrajectoryPercentages(config, DTModel, MainModel, trajectory_number=trajectory_number, loaded_set=loaded_set)
    else:
        return TrajectoryMeanError(config, DTModel, MainModel, trajectory_number = trajectory_number, loaded_set=loaded_set)

def Uniform(config, DTModel, MainModel, n = 10000, loaded_set = None):
    if config["surrogate"]["classifier"]:
        return UniformPercentages(config, DTModel, MainModel, n = n, loaded_set=loaded_set)
    else:
        return UniformMeanErrors(config, DTModel, MainModel, n = n, loaded_set=loaded_set)

def TrajectoryPercentages(config, DTModel, MainModel, trajectory_number = 1, loaded_set = None):
    if loaded_set == None:
        samples = trajectory_sampler(config, MainModel, n=trajectory_number)

    else:
        samples = loaded_set

    X = samples[0]
    #X = pd.DataFrame(X, columns=config["picture"]["labels"])
    out_DT = DTModel.Predict(X)
    Correct = 0
    Incorrect = 0
    for i in range(len(out_DT)):
        if out_DT[i] == samples[1][i]:
            Correct = Correct + 1
        else:
            Incorrect = Incorrect + 1
    total = len(out_DT)
    return {"Correct":Correct/total, "Incorrect":Incorrect/total}


def TrajectoryMeanError(config, DTModel, MainModel, trajectory_number=1, loaded_set = None):
    if loaded_set == None:
        samples = trajectory_sampler(config, MainModel, n=trajectory_number)

    else:
        samples = loaded_set
    X = samples[0]
    y_true = samples[1]
    #X         = pd.DataFrame(X, columns=config["picture"]["labels"])
    y_pred    = DTModel.Predict(X)        
    mse_sum   = 0.0                       
    mae_sum   = 0.0 
    med_sum   = 0.0                      
    for i in range(len(y_pred)):
        
        pred_vec = np.asarray(y_pred[i])
        true_vec = np.asarray(y_true[i])

        err_vec  = pred_vec - true_vec
        
        mse_sum += (1/err_vec.size)*np.sum(err_vec ** 2)     
        mae_sum += (1/err_vec.size)*np.sum(np.abs(err_vec))  
        med_sum += np.linalg.norm(err_vec)  


    mse = mse_sum / len(y_pred)
    mae = mae_sum / len(y_pred)
    med = med_sum / len(y_pred)
    return {"MSE": mse, "MAE": mae, "MED": med}

def UniformPercentages(config, DTModel, MainModel, n=100, loaded_set = None):
    if loaded_set == None:
        samples = uniform_sample_policy(config, MainModel, n=n)

    else:
        samples = loaded_set
    
    X = samples[0]
    #X = pd.DataFrame(X, columns=config["picture"]["labels"])
    out_DT = DTModel.Predict(X)
    Correct = 0
    Incorrect = 0
    for i in range(len(out_DT)):
        if out_DT[i] == samples[1][i]:
            Correct = Correct + 1
        else:
            Incorrect = Incorrect + 1
    total = len(out_DT)
    return {"Correct": Correct/total, "Incorrect":Incorrect/total}

def UniformMeanErrors(config, DTModel, MainModel, n=100, loaded_set = None):
    if loaded_set == None:
        samples = trajectory_sampler(config, MainModel, n=n)

    else:
        samples = loaded_set
    X = samples[0]
    y_true = samples[1]
    #X         = pd.DataFrame(X, columns=config["picture"]["labels"])
    y_pred    = DTModel.Predict(X)        
    mse_sum   = 0.0                       
    mae_sum   = 0.0 
    med_sum   = 0.0                      
    for i in range(len(y_pred)):
        
        pred_vec = np.asarray(y_pred[i])
        true_vec = np.asarray(y_true[i])

        err_vec  = pred_vec - true_vec
        
        mse_sum += (1/err_vec.size)*np.sum(err_vec ** 2)     
        mae_sum += (1/err_vec.size)*np.sum(np.abs(err_vec))  
        med_sum += np.linalg.norm(err_vec)  


    mse = mse_sum / len(y_pred)
    mae = mae_sum / len(y_pred)
    med = med_sum / len(y_pred)
    return {"MSE": mse, "MAE": mae, "MED": med}

def SingleEpisodicDivergence(config, DTModel, MainModel, seed, DEEP_SAR = None):
    runner_DT = GymRunner(config, None, seed = seed)
    
    if DEEP_SAR == None:
        runner_DEEP = GymRunner(config, MainModel, seed = seed)
        DEEP_SAR = runner_DEEP.runner(seed=seed, use_act=False)

    #This is bad programming practice, but 'use_dist' calls model.act. we turn that on for decision trees to get an
    #array of the same size as expected for an action. If it were turned on for a deep-policy, the policy would be stochastic

    DT_SAR = runner_DT.runner(model = DTModel, seed = seed, use_act=True)

    DT_states = DT_SAR["observation"]
    DT_actions = DT_SAR["action"]
    DT_reward = DT_SAR["reward"]

    DEEP_states = DEEP_SAR["observation"]
    DEEP_actions = DEEP_SAR["action"]
    DEEP_reward = DEEP_SAR["reward"]


    min_states = min(DT_states.shape[0], DEEP_states.shape[0])

    state_euclidean_distances = np.linalg.norm(DT_states[:min_states] - DEEP_states[:min_states], axis = 1)
    if config["surrogate"]["classifier"]:
        action_error = np.sum(DT_actions[:min_states] != DEEP_actions[:min_states])
    else:
        action_error = np.linalg.norm(DT_actions[:min_states] - DEEP_actions[:min_states], axis = 1)

    reward_differences =  DEEP_reward[:min_states] - DT_reward[:min_states]

    
    return {"StateEucDifferences": state_euclidean_distances, "ActionError": action_error, "RewardDifferences": reward_differences}

def EpisodeDivergence(config, DTModels, MainModel, n = 1):
    returner = []
    for i in range(n):
        seed = np.random.randint(1, 10001)
        runner_DEEP = GymRunner(config, MainModel, seed = seed)
        DEEP_SAR = runner_DEEP.runner(seed=seed, use_act=False)
        
        if DEEP_SAR == None:
            DEEP_SAR = runner_DEEP.runner(seed=seed, use_act=False)
        
        for model in DTModels:
            returner.append(SingleEpisodicDivergence(config,model,MainModel,seed,DEEP_SAR=DEEP_SAR))
        
        return returner