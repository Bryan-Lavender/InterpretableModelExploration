import numpy as np
import torch
import torch.nn as nn

def build_cnn_mlp(input_size, output_size, n_layers, size, cnn_settings):
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

    create_cnn(cnn_settings, modules)
    modules.append(nn.Flatten())
    
    conv_dims = calculate_conv_dims(cnn_settings, input_size)
    modules.append(nn.Linear(conv_dims, size))
    modules.append(nn.ReLU())

    for _ in range(n_layers - 1):
        modules.append(nn.Linear(size, size))
        modules.append(nn.ReLU())

   
    modules.append(nn.Linear(size, output_size))


    model = nn.Sequential(*modules)

    return model

def create_cnn(cnn_settings, modules):

    for i in range(len(cnn_settings)):

        convolution = nn.Conv2d(
            in_channels=cnn_settings[i]["in_channels"],
            out_channels=cnn_settings[i]["out_channels"],
            kernel_size=cnn_settings[i]["kernel_size"],
            stride=cnn_settings[i]["stride"]
        )

        modules.append(convolution)
        modules.append(nn.ReLU())

        
# NOTE: remember that you remobed to.device just to test

# define conv layers, pass the temp tensor through each layer, get size
def calculate_conv_dims(cnn_settings, input_size):

    temp = torch.zeros(1, *input_size)

    for i in range(len(cnn_settings)):

        convolution = nn.Conv2d(
            in_channels=cnn_settings[i]["in_channels"],
            out_channels=cnn_settings[i]["out_channels"],
            kernel_size=cnn_settings[i]["kernel_size"],
            stride=cnn_settings[i]["stride"]
        )

        temp = convolution(temp)

    return int(np.prod(temp.size()))

    