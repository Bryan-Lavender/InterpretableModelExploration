import numpy as np
import torch
import torch.nn as nn

conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1).to(device='cuda')
conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1).to(device='cuda')

def build_mlp(input_size, output_size, n_layers, size):
    """
    Builds a multi-layer perceptron in Pytorch based on a user's input

    Args:
        input_size (int): the dimension of inputs to be given to the network
        output_size (int): the dimension of the output
        n_layers (int): the number of hidden layers of the network
        size (int): the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.

    """

    if n_layers < 1:
        raise ValueError("n_layers must be at least 1")

    modules = []

    print(output_size)

    modules.append(conv1)
    modules.append(nn.ReLU())
    modules.append(conv2)
    modules.append(nn.ReLU())

    conv_dims = calculate_conv_dims(input_size)
    modules.append(nn.Flatten())

    modules.append(nn.Linear(62832, size))
    modules.append(nn.ReLU())

    for _ in range(n_layers - 1):
        modules.append(nn.Linear(size, size))
        modules.append(nn.ReLU())

   
    modules.append(nn.Linear(size, output_size))


    model = nn.Sequential(*modules)

    return model

# this is so we can correctly calculate size of the inputs for the actor critic networks
def calculate_conv_dims(input_size):
    state = torch.zeros(1, *input_size).to(device='cuda')

    x = conv1(state)
    x = conv2(x)

    return int(np.prod(x.size()))

    