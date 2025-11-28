#!/usr/bin/env python3
"""
Train an autoencoder on airfoil performance data.
Takes Reynolds, alpha, Cl, cd, L/D as input and compresses to 8 dimensions.
"""

import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import csv
import json

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
AIRFOIL_DATA_DIR = PROJECT_ROOT / "data" / "airfoil_data"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 8
INPUT_DIM = 101  # 1 (Reynolds) + 25 (alpha) + 25 (Cl) + 25 (cd) + 25 (L/D)
ALPHA_LENGTH = 25  # Number of alpha values (-10 to 14)
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class PerformanceDataset(Dataset):
    """Dataset for airfoil performance data."""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class PerformanceAutoencoder(nn.Module):
    """Autoencoder network: 101 -> 8 -> 101"""
    def __init__(self, input_dim=101, latent_dim=8):
        super(PerformanceAutoencoder, self).__init__()
        
        # Encoder: 101 -> 64 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        
        # Decoder: 8 -> 16 -> 32 -> 64 -> 101
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        """Encode input to latent space."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to output."""
        return self.decoder(z)

def load_performance_data():
    """Load all performance data from CSV files."""
    csv_files = sorted(AIRFOIL_DATA_DIR.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    failed = 0
    
    for i, csv_file in enumerate(csv_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(csv_files)} files...")
        
        try:
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)  # Skip header
                
                for row in reader:
                    if len(row) < 5:  # Need at least Reynolds, alpha, Cl, cd, L/D
                        continue
                    
                    try:
                        # Parse Reynolds
                        reynolds = float(row[0])
                        
                        # Parse JSON lists
                        alpha = json.loads(row[1])
                        cl = json.loads(row[2])
                        cd = json.loads(row[3])
                        l_d = json.loads(row[4])
                        
                        # Handle None values in L/D (from NaN)
                        l_d = [0.0 if x is None else float(x) for x in l_d]
                        
                        # Normalize Reynolds (log scale is more appropriate)
                        reynolds_log = np.log10(reynolds)
                        
                        # Ensure all lists have the same length
                        if len(alpha) == ALPHA_LENGTH and len(cl) == ALPHA_LENGTH and \
                           len(cd) == ALPHA_LENGTH and len(l_d) == ALPHA_LENGTH:
                            # Flatten: [Reynolds, alpha..., Cl..., cd..., L/D...]
                            flattened = [reynolds_log] + alpha + cl + cd + l_d
                            
                            if len(flattened) == INPUT_DIM:
                                all_data.append(flattened)
                    except (json.JSONDecodeError, ValueError, IndexError) as e:
                        continue
                        
        except Exception as e:
            failed += 1
            continue
    
    print(f"Successfully loaded {len(all_data)} performance data samples, {failed} files failed")
    return np.array(all_data)

def train_autoencoder():
    """Train the autoencoder model."""
    # Load data
    print("Loading performance data...")
    data = load_performance_data()
    
    if len(data) == 0:
        raise ValueError("No performance data loaded!")
    
    print(f"Data shape: {data.shape}")
    
    # Normalize data
    print("Normalizing data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Save scaler for later use
    MODELS_DIR.mkdir(exist_ok=True)
    scaler_path = MODELS_DIR / "xfoil_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(data_normalized, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_data)}, Validation set: {len(val_data)}")
    
    # Create data loaders
    train_dataset = PerformanceDataset(train_data)
    val_dataset = PerformanceDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = PerformanceAutoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15)
    
    print(f"\nModel architecture:")
    print(f"  Input: {INPUT_DIM} dimensions")
    print(f"  Latent: {LATENT_DIM} dimensions")
    print(f"  Output: {INPUT_DIM} dimensions")
    print(f"\nStarting training for {EPOCHS} epochs...")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(DEVICE)
                output = model(batch)
                loss = criterion(output, batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODELS_DIR / "xfoil_autoencoder_best.pth")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "xfoil_autoencoder_best.pth"))
    
    # Save encoder and decoder separately
    print("\nSaving encoder and decoder...")
    
    # Create separate encoder and decoder models
    encoder = nn.Sequential(*list(model.encoder.children())).to(DEVICE)
    decoder = nn.Sequential(*list(model.decoder.children())).to(DEVICE)
    
    # Save full models (architecture + weights) for easier loading
    encoder.eval()
    encoder_path = MODELS_DIR / "xfoil_encoder.pth"
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'input_dim': INPUT_DIM,
        'latent_dim': LATENT_DIM,
        'architecture': 'encoder'
    }, encoder_path)
    print(f"Saved encoder to {encoder_path}")
    
    decoder.eval()
    decoder_path = MODELS_DIR / "xfoil_decoder.pth"
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'input_dim': LATENT_DIM,
        'output_dim': INPUT_DIM,
        'architecture': 'decoder'
    }, decoder_path)
    print(f"Saved decoder to {decoder_path}")
    
    # Save full autoencoder for reference
    autoencoder_path = MODELS_DIR / "xfoil_autoencoder.pth"
    torch.save(model.state_dict(), autoencoder_path)
    print(f"Saved full autoencoder to {autoencoder_path}")
    
    print("\nAll models saved successfully!")

if __name__ == "__main__":
    train_autoencoder()

