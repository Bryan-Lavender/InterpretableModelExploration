# import torch
# from torch import nn

# class TwoDWeights_LRPModel(nn.Module):
#     def __init__(self, model, top_k=5):
#         super().__init__()
#         self.model = model
#         self.top_k = top_k

#         self.model.eval()
#         self.layers = self._get_layers()[::-1]  # Extract only fully connected layers
#         self.activations = []  # List to store activations during forward pass
    
#     def _get_layers(self):
#         """
#         Retrieves the fully connected layers (nn.Linear) of the model.
#         """
#         layers = []
#         # Recursively collect layers if model has nested structures
#         for layer in self.model.children():
#             if isinstance(layer, nn.Linear):  # Adjust if other types like ReLU are involved
#                 layers.append(layer)
#             elif len(list(layer.children())) > 0:
#                 layers.extend(self._get_layers_from_module(layer))
#         return layers
    
#     def _get_layers_from_module(self, module):
#         """Recursively retrieves fully connected layers from nested modules."""
#         layers = []
#         for layer in module.children():
#             if isinstance(layer, nn.Linear):
#                 layers.append(layer)
#             elif len(list(layer.children())) > 0:
#                 layers.extend(self._get_layers_from_module(layer))
#         return layers
    
#     def _register_hooks(self):
#         """
#         Registers forward hooks to capture activations from each fully connected layer during the forward pass.
#         """
#         self.activations = []  # Reset activations list
#         for layer in self.layers:
#             layer.register_forward_hook(self._hook_fn)

#     def _hook_fn(self, module, input, output):
#         """
#         Hook function to capture activations from each layer.
#         """
        
#         self.activations.append(input[0].clone())  # Store the output of each layer
    
#     def forward(self, x):
#         self._register_hooks()  # Register hooks before the forward pass
#         output = self.model(x)  # Standard forward pass
#         return output
        
#     def propagate_relevance(self, relevance, layer, activations):
#         """
#         Propagates the relevance scores backward using the Z-Rule.
#         """
#         weights = layer.weight
#         weights = weights.T
#         biases = layer.bias
#         z = []
#         CorVec = []
        
#         for i in range(weights.shape[1]):
#             tmp = activations*weights[:,i] + biases[i]
#             z.append(tmp.sum())
#             CorVec.append(tmp)
        
#         z = torch.stack(z)
#         #z += 1e-9 * (z >= 0).float() + (-1e-9) * (z < 0).float()
#         z += 1e-9
#         CorVec = torch.stack(CorVec)


#         normalized_CorVec = CorVec / z.unsqueeze(1)
        
        
#         relevance_scores = relevance @ normalized_CorVec
        
#         return relevance_scores
    
#     def backward_relevance_propagation(self, output):
#         """
#         Applies backward relevance propagation using captured activations and layers.
        
#         Parameters:
#         - output: The output from the forward pass.
        
#         Returns:
#         - input_relevance: Relevance scores attributed to the input layer.
#         """
#         relevance = output.clone()  # Start with the output layer relevance
        
#         # Propagate relevance backward using the layers and activations
#         # print(self.layers)
#         # biases = [i.bias for i in self.layers]
#         # print(self.activations)
#         # print(biases)
#         # print([i for i in self.model.modules()])
#         vectors = []
#         input_relevance = []
#         with torch.no_grad():
#             for i in range(len(relevance)):
#                 vector = torch.zeros_like(relevance)  # Create a zero vector of the same shape
#                 vector[i] = relevance[i]  # Set the i-th position to the corresponding relevance value
#                 vectors.append(vector)

#             # Display the generated vectors
#             for v in vectors:
#                 relevance = v
#                 for layer, activation in zip(self.layers, self.activations[::-1]):
#                     relevance = self.propagate_relevance(relevance, layer, activation)
                
#                 input_relevance.append(relevance)
#         self.activations = []
#         return input_relevance


#     def get_FI(self, point, printer = False):
#         if printer:
#             print("point", point)
#             out = self.forward(point)
#             print("output",out)
#             rel = self.backward_relevance_propagation(out)
#             print("relevence", rel)
#             return out,rel
        
#         out = self.forward(point)
#         return out,self.backward_relevance_propagation(out)
    

import torch
import torch.nn as nn
import random
import numpy as np

# Set deterministic behavior
def set_deterministic(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic()

class TwoDWeights_LRPModel(nn.Module):
    def __init__(self, model, top_k=5):
        super().__init__()
        self.model = model.eval()  # Ensure model is in eval mode
        self.top_k = top_k
        self.layers = self._get_layers()[::-1]  # Get fully connected layers in reverse order
        self.activations = []
        self._hook_handles = []

    def _get_layers(self):
        """Retrieve fully connected (nn.Linear) layers from model."""
        layers = []
        for layer in self.model.children():
            if isinstance(layer, nn.Linear):
                layers.append(layer)
            elif len(list(layer.children())) > 0:
                layers.extend(self._get_layers_from_module(layer))
        return layers

    def _get_layers_from_module(self, module):
        """Helper for nested module traversal."""
        layers = []
        for layer in module.children():
            if isinstance(layer, nn.Linear):
                layers.append(layer)
            elif len(list(layer.children())) > 0:
                layers.extend(self._get_layers_from_module(layer))
        return layers

    def _register_hooks(self):
        """Attach forward hooks to record activations."""
        self.activations = []
        self._hook_handles = []
        for layer in self.layers:
            handle = layer.register_forward_hook(self._hook_fn)
            self._hook_handles.append(handle)

    def _remove_hooks(self):
        """Remove all forward hooks after forward pass."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def _hook_fn(self, module, input, output):
        """Store pre-activation input for each layer."""
        self.activations.append(input[0].clone())

    def forward(self, x):
        """Run forward pass and register hooks."""
        self._register_hooks()
        with torch.no_grad():
            output = self.model(x)
        self._remove_hooks()
        return output

    def propagate_relevance(self, relevance, layer, activations):
        """Z-Rule propagation: backward relevance propagation."""
        weights = layer.weight.T
        biases = layer.bias
        z = []
        CorVec = []

        for i in range(weights.shape[1]):
            tmp = activations * weights[:, i] 
            z.append(tmp.sum())
            CorVec.append(tmp)

        z = torch.stack(z)
        epsilon = 1e-9 * z.sign()  # Stabilized division
        z += epsilon
        CorVec = torch.stack(CorVec)
        normalized_CorVec = CorVec / z.unsqueeze(1)
        relevance_scores = relevance @ normalized_CorVec
        return relevance_scores

    def backward_relevance_propagation(self, output):
        """Propagate relevance scores from output back to input."""
        relevance = output.clone()
        input_relevance = []

        with torch.no_grad():
            for i in range(len(relevance)):
                vector = torch.zeros_like(relevance)
                vector[i] = relevance[i]
                rel = vector
                for layer, activation in zip(self.layers, self.activations[::-1]):
                    rel = self.propagate_relevance(rel, layer, activation)
                input_relevance.append(rel)

        self.activations = []
        return input_relevance

    def get_FI(self, point, printer=False):
        """Perform full relevance analysis on input `point`."""
        if printer:
            print("point:", point)
        out = self.forward(point)
        if printer:
            print("output:", out)
        rel = self.backward_relevance_propagation(out)
        if printer:
            print("relevance:", rel)
        return out, rel