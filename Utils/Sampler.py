import torch
import numpy as np
from .EnvRunner import GymRunner
def uniform_sample_policy(config, model, n, dtype = torch.float32):
    device = None
    bounds = config["sampler"]["bounds"]
    if torch.cuda.is_available():
        device="cuda"
    else:
        device="cpu"
    b = torch.as_tensor(bounds, dtype=dtype, device=device)
    lows, highs = b[0], b[1]

    # Validate
    if (highs < lows).any():
        raise ValueError("Upper bound is smaller than lower bound in at least one dimension.")

    # Draw in [0,1) → scale to [low, high]
    r = torch.rand(n, lows.numel(), dtype=dtype, device=device)
    in_x =  (r * (highs - lows) + lows)

    out_y = model.policy.network.forward(in_x)
    if config["surrogate"]["classifier"]:
        out_class = torch.argmax(out_y, axis=1)
        return(in_x.detach().numpy(), out_class.detach().numpy(), out_y.detach().numpy())
    else:
        return(in_x.detach().numpy(), out_y.detach().numpy())
def execution_obs_sample(config, model, n):
    env_runner = GymRunner(config, model)
    state_action = env_runner.runner()
    
    while len(state_action["observation"])< n:
        new_state_action = env_runner.runner()
        state_action["observation"] = np.concatenate((state_action["observation"],new_state_action["observation"]))
        state_action["reward"] = np.concatenate((state_action["reward"],new_state_action["reward"]))
        state_action["action"] = np.concatenate((state_action["action"],new_state_action["action"]))
    indicies = np.random.choice(range(len(state_action["observation"])), size = n)
    if "activations" in state_action.keys():
        return(state_action["observation"][indicies], state_action["action"][indicies], state_action["activations"][indicies])
    return(state_action["observation"][indicies], state_action["action"][indicies])

def trajectory_sampler(config, model, n, k=None, get_activations = False, use_seed = False):
    env_runner = GymRunner(config, model)
    state_action = env_runner.runner(return_activations=get_activations)

    
    for i in range(n-1):
        if use_seed:
            seed = np.random.randint(10001, high = 100000)
            new_state_action = env_runner.runner(seed=seed,return_activations=get_activations)
        else:
            new_state_action = env_runner.runner(return_activations=get_activations)
        state_action["observation"] = np.concatenate((state_action["observation"],new_state_action["observation"]))
        state_action["reward"] = np.concatenate((state_action["reward"],new_state_action["reward"]))
        state_action["action"] = np.concatenate((state_action["action"],new_state_action["action"]))
        if "activations" in new_state_action.keys():
            state_action["activations"] = np.concatenate((state_action["activations"],new_state_action["activations"]))
    if k != None:
        indicies = np.random.choice(range(len(state_action["observation"])), size = k)
        return(state_action["observation"][indicies], state_action["action"][indicies])

    else:
        if "activations" in state_action.keys():
            return(state_action["observation"], state_action["action"], state_action["activations"])
        return(state_action["observation"], state_action["action"])
    
