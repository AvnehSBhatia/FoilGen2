#!/usr/bin/env python3
"""
Train an autoencoder on airfoil coordinate data.
Compresses 400 points (200x + 200y) to 16 dimensions.
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

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
BIGFOIL_DIR = PROJECT_ROOT / "data" / "bigfoil"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 16
INPUT_DIM = 400  # 200 x-coordinates + 200 y-coordinates
TARGET_POINTS = 200  # Number of points per airfoil
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0005
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

class AirfoilDataset(Dataset):
    """Dataset for airfoil coordinates."""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class Autoencoder(nn.Module):
    """Autoencoder network: 400 -> 16 -> 400"""
    def __init__(self, input_dim=400, latent_dim=16):
        super(Autoencoder, self).__init__()
        
        # Encoder: 400 -> 128 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: 16 -> 32 -> 64 -> 128 -> 400
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, input_dim)
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

def parse_airfoil_file(filepath):
    """Parse airfoil coordinate file and return x, y coordinates."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Skip first line (name/description)
    coords = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        try:
            parts = line.split()
            if len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
        except ValueError:
            continue
    
    if len(coords) < 2:
        return None, None
    
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]


def load_all_airfoils():
    """Load and preprocess all airfoil files."""
    dat_files = sorted(BIGFOIL_DIR.glob("*.dat"))
    print(f"Found {len(dat_files)} airfoil files")
    
    all_data = []
    failed = 0
    
    for i, dat_file in enumerate(dat_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(dat_files)} files...")
        
        try:
            x_coords, y_coords = parse_airfoil_file(dat_file)
            if x_coords is None:
                failed += 1
                continue
            
            # Check that we have exactly 200 points
            if len(x_coords) != TARGET_POINTS or len(y_coords) != TARGET_POINTS:
                failed += 1
                continue
            
            # Flatten: [x1, x2, ..., x200, y1, y2, ..., y200]
            flattened = np.concatenate([x_coords, y_coords])
            
            if len(flattened) == INPUT_DIM:
                all_data.append(flattened)
            else:
                failed += 1
        except Exception as e:
            failed += 1
            continue
    
    print(f"Successfully loaded {len(all_data)} airfoils, {failed} failed")
    return np.array(all_data)

def train_autoencoder():
    """Train the autoencoder model."""
    # Load data
    print("Loading airfoil data...")
    data = load_all_airfoils()
    
    if len(data) == 0:
        raise ValueError("No airfoil data loaded!")
    
    print(f"Data shape: {data.shape}")
    
    # Normalize data
    print("Normalizing data...")
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    
    # Save scaler for later use
    MODELS_DIR.mkdir(exist_ok=True)
    scaler_path = MODELS_DIR / "airfoil_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Saved scaler to {scaler_path}")
    
    # Split into train and validation sets
    train_data, val_data = train_test_split(data_normalized, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_data)}, Validation set: {len(val_data)}")
    
    # Create data loaders
    train_dataset = AirfoilDataset(train_data)
    val_dataset = AirfoilDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = Autoencoder(input_dim=INPUT_DIM, latent_dim=LATENT_DIM).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
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
            torch.save(model.state_dict(), MODELS_DIR / "airfoil_autoencoder_best.pth")
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "airfoil_autoencoder_best.pth"))
    
    # Save encoder and decoder separately
    print("\nSaving encoder and decoder...")
    
    # Create separate encoder and decoder models
    encoder = nn.Sequential(*list(model.encoder.children())).to(DEVICE)
    decoder = nn.Sequential(*list(model.decoder.children())).to(DEVICE)
    
    # Save full models (architecture + weights) for easier loading
    encoder.eval()
    encoder_path = MODELS_DIR / "airfoil_encoder.pth"
    torch.save({
        'model_state_dict': encoder.state_dict(),
        'input_dim': INPUT_DIM,
        'latent_dim': LATENT_DIM,
        'architecture': 'encoder'
    }, encoder_path)
    print(f"Saved encoder to {encoder_path}")
    
    decoder.eval()
    decoder_path = MODELS_DIR / "airfoil_decoder.pth"
    torch.save({
        'model_state_dict': decoder.state_dict(),
        'input_dim': LATENT_DIM,
        'output_dim': INPUT_DIM,
        'architecture': 'decoder'
    }, decoder_path)
    print(f"Saved decoder to {decoder_path}")
    
    # Save full autoencoder for reference
    autoencoder_path = MODELS_DIR / "airfoil_autoencoder.pth"
    torch.save(model.state_dict(), autoencoder_path)
    print(f"Saved full autoencoder to {autoencoder_path}")
    
    print("\nAll models saved successfully!")

if __name__ == "__main__":
    train_autoencoder()

