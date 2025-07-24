import numpy as np
import torch

def Q_diff(Activations):
    activations = torch.tensor(Activations.to_numpy())
    diffs = []
    for i in activations:
        dis = torch.distributions.Categorical(logits = i)
        is_min = None
        is_max = None
        for j in range(len(i)):
            log_prob = dis.log_prob(torch.tensor(j))
            if is_min == None or log_prob < is_min:
                is_min = log_prob
            if is_max == None or log_prob > is_max:
                is_max = log_prob
        diff = is_max - is_min
        diffs.append(diff.detach().numpy())
    diffs = np.stack(diffs)
    return diffs
