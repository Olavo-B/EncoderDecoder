'''
/***********************************************
 * File: decoder.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2025-01-06
 * License: [License Type]
 * Description: Decoder module for a Variational Autoencoder (VAE) in PyTorch.
 ***********************************************/
 '''

import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, context_dim, hidden_dims, output_dim):
        super(Decoder, self).__init__()
        layers = []
        dims = [context_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # No activation in the final layer
                layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)
