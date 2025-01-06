'''
/***********************************************
 * File: train.py
 * Author: Olavo Alves Barros Silva
 * Contact: olavo.barros@ufv.com
 * Date: 2025-01-06
 * License: [License Type]
 * Description: Train script for a Variational Autoencoder (VAE) in PyTorch.
 ***********************************************/
 '''

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.data_utils import load_data, normalize_data
from models.encoder import Encoder
from models.decoder import Decoder



def train_model(data_path, model_save_path, input_dim, context_dim, epochs, batch_size, learning_rate):
    # Load Dataset
    dataset = load_data(data_path)
    dataloader = DataLoader(dataset,transform=normalize_data, 
                            batch_size=batch_size, shuffle=True)

    # Initialize Encoder and Decoder Models
    encoder = Encoder(input_dim=10, hidden_dims=[64, 32], context_dim=4)
    decoder = Decoder(context_dim=4, hidden_dims=[32, 64], output_dim=10)
    criterion = nn.MSELoss()  # Example loss function
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    # Training Loop
    encoder.train()
    decoder.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(dataloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch + 1}/{epochs}")
            for batch in tepoch:
                inputs = batch

                # Forward Pass
                context_vector = encoder(inputs)
                outputs = decoder(context_vector)
                loss = criterion(outputs, inputs)

                # Backward Pass
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

    # Save Models
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'dataloaders': dataloader.state_dict()
    }, model_save_path)
    print(f"Models saved to {model_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an Encoder-Decoder MLP model")
    parser.add_argument("--data", type=str, required=True, help="Path to training data")
    parser.add_argument("--model", type=str, required=True, help="Path to save the trained models")
    parser.add_argument("--input_dim", type=int, default=10, help="Input dimension of the data")
    parser.add_argument("--context_dim", type=int, default=2, help="Dimension of the context vector")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")

    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_save_path=args.model,
        input_dim=args.input_dim,
        context_dim=args.context_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
