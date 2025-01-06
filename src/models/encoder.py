'''
/***********************************************
 * File: encoder.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2025-01-06
 * License: [License Type]
 * Description: Encoder module for a Variational Autoencoder (VAE) in PyTorch.
 ***********************************************/
 '''

import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, context_dim):
        super(Encoder, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [context_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation in the final layer
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)
