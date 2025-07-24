import numpy as np
import pandas as pd

def samples_to_pandas(config, samples_p):
    X = samples_p[0]
    X = pd.DataFrame(X, columns=config["picture"]["labels"])
    if config["surrogate"]["classifier"]:
        Y = pd.DataFrame(samples_p[1], columns = ["out"])
    else:
        samples_p
        Y = pd.DataFrame(samples_p[1], columns = config["picture"]["class_names"])
    if len(samples_p) == 3:
        activations = pd.DataFrame(samples_p[2], columns = config["picture"]["class_names"])
        return (X, Y, activations)
    return(X,Y)