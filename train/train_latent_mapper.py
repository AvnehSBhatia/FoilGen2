#!/usr/bin/env python3
"""
Train a model to map from xfoil_latent_vector to airfoil_latent_vector.
Takes 8D xfoil latent vector (performance) and outputs 8D airfoil latent vector (shape).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import csv
import json
import pickle

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
LATENT_VECTORS_CSV = PROJECT_ROOT / "data" / "latent_vectors.csv"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 16
BATCH_SIZE = 64
EPOCHS = 300
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class LatentMappingDataset(Dataset):
    """Dataset for xfoil -> airfoil latent vector mapping."""
    def __init__(self, xfoil_vectors, airfoil_vectors):
        self.xfoil_vectors = torch.FloatTensor(xfoil_vectors)
        self.airfoil_vectors = torch.FloatTensor(airfoil_vectors)
    
    def __len__(self):
        return len(self.xfoil_vectors)
    
    def __getitem__(self, idx):
        return self.xfoil_vectors[idx], self.airfoil_vectors[idx]

class LatentMapper(nn.Module):
    """Neural network to map xfoil latent vector to airfoil latent vector."""
    def __init__(self, input_dim=8, output_dim=8, hidden_dims=[128,256,128]):
        super(LatentMapper, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x): return self.network(x)

def load_latent_vectors():
    """Load xfoil and airfoil latent vectors from CSV."""
    print(f"Loading latent vectors from {LATENT_VECTORS_CSV}...")
    
    xfoil_vectors = []
    airfoil_vectors = []
    
    with open(LATENT_VECTORS_CSV, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        
        # Find column indices
        try:
            xfoil_idx = header.index('xfoil_latent_vector')
            airfoil_idx = header.index('airfoil_latent_vector')
        except ValueError as e:
            raise ValueError(f"Missing required columns in CSV: {e}")
        
        for row in reader:
            try:
                xfoil_latent = json.loads(row[xfoil_idx])
                airfoil_latent = json.loads(row[airfoil_idx])
                
                # Convert to floats and ensure correct length
                xfoil_latent = [float(x) for x in xfoil_latent]
                airfoil_latent = [float(x) for x in airfoil_latent]
                
                if len(xfoil_latent) == LATENT_DIM and len(airfoil_latent) == LATENT_DIM:
                    xfoil_vectors.append(xfoil_latent)
                    airfoil_vectors.append(airfoil_latent)
            except (json.JSONDecodeError, ValueError, IndexError) as e:
                continue
    
    print(f"Loaded {len(xfoil_vectors)} latent vector pairs")
    return np.array(xfoil_vectors), np.array(airfoil_vectors)

def train_mapper():
    """Train the latent vector mapping model."""
    # Load data
    xfoil_vectors, airfoil_vectors = load_latent_vectors()
    
    if len(xfoil_vectors) == 0:
        raise ValueError("No latent vector pairs loaded!")
    
    print(f"Data shapes: xfoil={xfoil_vectors.shape}, airfoil={airfoil_vectors.shape}")
    
    # Normalize input (xfoil vectors)
    print("Normalizing xfoil latent vectors...")
    xfoil_scaler = StandardScaler()
    xfoil_vectors_normalized = xfoil_scaler.fit_transform(xfoil_vectors)
    
    # Normalize output (airfoil vectors)
    print("Normalizing airfoil latent vectors...")
    airfoil_scaler = StandardScaler()
    airfoil_vectors_normalized = airfoil_scaler.fit_transform(airfoil_vectors)
    
    # Save scalers
    MODELS_DIR.mkdir(exist_ok=True)
    with open(MODELS_DIR / "latent_mapper_xfoil_scaler.pkl", "wb") as f:
        pickle.dump(xfoil_scaler, f)
    print(f"Saved xfoil scaler to {MODELS_DIR / 'latent_mapper_xfoil_scaler.pkl'}")
    
    with open(MODELS_DIR / "latent_mapper_airfoil_scaler.pkl", "wb") as f:
        pickle.dump(airfoil_scaler, f)
    print(f"Saved airfoil scaler to {MODELS_DIR / 'latent_mapper_airfoil_scaler.pkl'}")
    
    # Split into train and validation sets
    train_x, val_x, train_y, val_y = train_test_split(
        xfoil_vectors_normalized, airfoil_vectors_normalized, 
        test_size=0.2, random_state=42
    )
    print(f"Train set: {len(train_x)}, Validation set: {len(val_x)}")
    
    # Create data loaders
    train_dataset = LatentMappingDataset(train_x, train_y)
    val_dataset = LatentMappingDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = LatentMapper(input_dim=LATENT_DIM, output_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    print(f"\nModel architecture:")
    print(f"  Input: {LATENT_DIM} dimensions (xfoil latent)")
    print(f"  Output: {LATENT_DIM} dimensions (airfoil latent)")
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for xfoil_batch, airfoil_batch in train_loader:
            xfoil_batch = xfoil_batch.to(DEVICE)
            airfoil_batch = airfoil_batch.to(DEVICE)
            
            optimizer.zero_grad()
            output = model(xfoil_batch)
            loss = criterion(output, airfoil_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xfoil_batch, airfoil_batch in val_loader:
                xfoil_batch = xfoil_batch.to(DEVICE)
                airfoil_batch = airfoil_batch.to(DEVICE)
                
                output = model(xfoil_batch)
                loss = criterion(output, airfoil_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    
    # Save final model
    print("\nSaving final model...")
    model.eval()
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': LATENT_DIM,
        'output_dim': LATENT_DIM,
        'architecture': 'latent_mapper'
    }, MODELS_DIR / "latent_mapper.pth")
    print(f"Saved model to {MODELS_DIR / 'latent_mapper.pth'}")

if __name__ == "__main__":
    train_mapper()

