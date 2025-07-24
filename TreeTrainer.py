from Utils.Sampler import trajectory_sampler
from Utils.LoadModel import read_config, load_weights
from FeatureImportance.FI import FeatureImportance
import time
import numpy as np
import pickle
import json
import os
from Utils.Metrics import Trajectory, Uniform, EpisodeDivergence
from torch.utils.tensorboard import SummaryWriter

from DecisionTrees.Lavender_DT.DecisionTree import WeightedDecisionTrees
from DecisionTrees.Feature_specific_trees.DecisionTree import FeatureSpecifcTrees
from DecisionTrees.viper.viper import VIPER_reSampled, VIPER_weighted
from DecisionTrees.ScikitlearnDT.scikitlearnDT import SKLTree
def create_tree_config(config):
    if not config["env"]["discrete"]:
        config["Tree"] = {"criterion": "MSE", 
            "leaf_creator": "STD", 
            "splitting_function": "ImportanceWeighing",
            "weighing_method": "Var_Weighted",
            "object_names": None}
    else:
        config["Tree"] = {"criterion": "entropy", 
            "leaf_creator": "single_class", 
            "splitting_function": "ImportanceWeighing",
            "weighing_method": "Var_Weighted",
            "object_names": None}
        
def get_static_metrics(DOMAIN, baseline, path):
    Set_X = np.load("Test_Sets/Uniform_Samples/" + DOMAIN + "/" + "X.npy")
    Set_Y = np.load("Test_Sets/Uniform_Samples/" + DOMAIN + "/" + "Y.npy")
    Uniform_metric =  Uniform(config, baseline, model, loaded_set=(Set_X, Set_Y)), 
    #ram, possibly
    
    Set_X = np.load("Test_Sets/Trajectory_Samples/" + DOMAIN + "/" + "X.npy")
    Set_Y = np.load("Test_Sets/Trajectory_Samples/" + DOMAIN + "/" + "Y.npy")
    Trajectory_metric = Trajectory(config, baseline, model, loaded_set = (Set_X, Set_Y))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"Uniform": Uniform_metric, "Trajector": Trajectory_metric}
    with open(path, "w") as f:
        json.dump(obj, f)
    
#lets begin with classification domains
domains = ["cartpole", "acrobot", "lunar_lander"]
times = {}
times_all = {}
DOMAIN = "cartpole"
config = read_config(DOMAIN)
model = load_weights(config)
create_tree_config(config)
weighing_type = "normal"
for traj_num in [1,2,3,4,5,6,7,8,9,10]:
    for run in range(100):
        X,Y, Activations = trajectory_sampler(config, model, n = traj_num, get_activations=True)
        out = None
        FI = None
        config["Tree"] = {"criterion": "entropy", 
                "leaf_creator": "single_class", 
                "splitting_function": "normal",
                "weighing_method": weighing_type,
                "object_names": None}

        tree_type = "weighted_" + weighing_type
        print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
        if tree_type not in times.keys():
            times[tree_type] = []
        tree = WeightedDecisionTrees(config)
        start = time.time()
        tree.fit(X,Y,FI,out)
        path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
        tree.save(path_str)
        times[tree_type].append(time.time() - start)
        path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

        metrics = get_static_metrics(DOMAIN, tree, path_str)
exit()
for DOMAIN in domains:
    config = read_config(DOMAIN)
    model = load_weights(config)
    create_tree_config(config)
    #increase trajectories from 1 to 100
    for traj_num in [1,2,3,4,5,6,7,8,9,10]:
        #save results of 100 trees made this way

        for run in range(100): 
            
            X,Y, Activations = trajectory_sampler(config, model, n = traj_num, get_activations=True)
            FI_calculator = FeatureImportance("FD", model.network)
            out, FI = FI_calculator.Relevence(X)

            #scikit learn trees
            
            tree_type = "baseline"
            print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
            if tree_type not in times.keys():
                times[tree_type] = []
            baseline = SKLTree(config)
            start = time.time()
            baseline.fit(X,Y)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
            baseline.save(path_str)
            times[tree_type].append(time.time() - start)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

            metrics = get_static_metrics(DOMAIN, baseline, path_str)
            
            
            start = time.time()

            #VIPER_resample
            tree_type = "VIPER_resample"
            print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
            if tree_type not in times.keys():
                times[tree_type] = []
            VIPER_re = VIPER_reSampled(config)
            start = time.time()
            VIPER_re.train_viper_categorical(X,Y,Activations)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
            VIPER_re.save(path_str)
            times[tree_type].append(time.time() - start)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

            metrics = get_static_metrics(DOMAIN, VIPER_re, path_str)
            

            #VIPER_resample
            tree_type = "VIPER_weigh"
            print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
            if tree_type not in times.keys():
                times[tree_type] = []
            VIPER_weigh = VIPER_weighted(config)
            start = time.time()
            VIPER_weigh.train_viper_categorical(X,Y,Activations)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
            VIPER_weigh.save(path_str)
            times[tree_type].append(time.time() - start)
            path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

            metrics = get_static_metrics(DOMAIN, VIPER_weigh, path_str)

            #Weighted DecisionTrees
            weighing_types = ["Var_Weighted", "Max_Avg", "Max_All", "Double_Avg", "Class"]
            for weighing_type in weighing_types:

                config["Tree"] = {"criterion": "entropy", 
                "leaf_creator": "single_class", 
                "splitting_function": "ImportanceWeighing",
                "weighing_method": weighing_type,
                "object_names": None}

                tree_type = "weighted_" + weighing_type
                print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
                if tree_type not in times.keys():
                    times[tree_type] = []
                tree = WeightedDecisionTrees(config)
                start = time.time()
                tree.fit(X,Y,FI,out)
                path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
                tree.save(path_str)
                times[tree_type].append(time.time() - start)
                path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

                metrics = get_static_metrics(DOMAIN, tree, path_str)
            
            #feature_chosen
            weighing_types = ["Var_Weighted", "Max_Avg", "Max_All", "Double_Avg", "Class"]
            for weighing_type in weighing_types:

                config["Tree"] = {"criterion": "entropy", 
                "leaf_creator": "single_class", 
                "splitting_function": "ImportanceWeighing",
                "weighing_method": weighing_type,
                "object_names": None}

                tree_type = "FIselected_" + weighing_type
                print(DOMAIN + " traj_num: " + str(traj_num) + " run: " + str(run) + " ~ " + tree_type)
                if tree_type not in times.keys():
                    times[tree_type] = []
                tree = FeatureSpecifcTrees(config)
                start = time.time()
                tree.fit(X,Y,FI,out)
                path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/tree/" + str(run) + ".pkl"
                tree.save(path_str)
                times[tree_type].append(time.time() - start)
                path_str = "Runs/" + DOMAIN + "/"+tree_type+"/" + "Traj_Num_" + str(traj_num) + "/static_metrics/" + str(run) + ".json"

                metrics = get_static_metrics(DOMAIN, tree, path_str)
        
        times_all[DOMAIN + "_" + str(traj_num)] = times 

with open("RunTimes.json", "wb") as f:
    json.dump(times_all, f)

