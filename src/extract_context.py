'''
/***********************************************
 * File: extract_context.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2025-01-06
 * License: [License Type]
 * Description: Extracts context vectors from an encoder and visualizes them using PCA or t-SNE.
 ***********************************************/
 '''

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def extract_context_vectors(encoder, dataloader):
    """Extracts context vectors from the encoder for the entire dataset."""
    context_vectors = []
    encoder.eval()
    with torch.no_grad():
        for batch in dataloader:
            context = encoder(batch)
            context_vectors.append(context.numpy())
    return np.vstack(context_vectors)


def visualize_context_vectors(context_vectors, method="pca"):
    """Visualizes context vectors using PCA or t-SNE."""
    if method == "pca":
        reducer = PCA(n_components=2)
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    reduced_context = reducer.fit_transform(context_vectors)
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_context[:, 0], reduced_context[:, 1], alpha=0.7, s=20, cmap='viridis')
    plt.title(f"Context Vectors Visualization ({method.upper()})")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
