#!/usr/bin/env python3
"""
Test script for the trained autoencoder.
Tests encoding and decoding of airfoil coordinates.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
BIGFOIL_DIR = PROJECT_ROOT / "data" / "bigfoil"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 16
INPUT_DIM = 400  # 200 x-coordinates + 200 y-coordinates
TARGET_POINTS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

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

def load_encoder_decoder():
    """Load the saved encoder and decoder models."""
    # Load encoder
    encoder_checkpoint = torch.load(MODELS_DIR / "airfoil_encoder.pth", map_location=DEVICE)
    encoder = nn.Sequential(
        nn.Linear(encoder_checkpoint['input_dim'], 256),
        nn.Tanh(),
        nn.Linear(256, 128),
        nn.Tanh(),
        nn.Linear(128, 32),
        nn.Tanh(),
        nn.Linear(32, encoder_checkpoint['latent_dim'])
    ).to(DEVICE)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    
    # Load decoder
    decoder_checkpoint = torch.load(MODELS_DIR / "airfoil_decoder.pth", map_location=DEVICE)
    decoder = nn.Sequential(
        nn.Linear(decoder_checkpoint['input_dim'], 32),
        nn.Tanh(),
        nn.Linear(32, 128),
        nn.Tanh(),
        nn.Linear(128, 256),
        nn.Tanh(),
        nn.Linear(256, decoder_checkpoint['output_dim'])
    ).to(DEVICE)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder.eval()
    
    print("✓ Encoder and decoder loaded successfully")
    return encoder, decoder, encoder_checkpoint['input_dim'], decoder_checkpoint['output_dim']

def load_scaler():
    """Load the data scaler."""
    with open(MODELS_DIR / "airfoil_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
    return scaler

def test_autoencoder():
    """Test the autoencoder on sample airfoils."""
    # Check if models exist
    if not (MODELS_DIR / "airfoil_encoder.pth").exists() or not (MODELS_DIR / "airfoil_decoder.pth").exists():
        print("Error: Model files not found. Please train the autoencoder first.")
        return
    
    if not (MODELS_DIR / "airfoil_scaler.pkl").exists():
        print("Error: Scaler file not found. Please train the autoencoder first.")
        return
    
    # Load models and scaler
    encoder, decoder, input_dim, output_dim = load_encoder_decoder()
    scaler = load_scaler()
    
    print(f"\nModel architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Output dimension: {output_dim}")
    
    # Get some test airfoils
    dat_files = sorted(BIGFOIL_DIR.glob("*.dat"))[:5]  # Test on first 5 airfoils
    print(f"\nTesting on {len(dat_files)} airfoils...")
    
    all_errors = []
    
    for i, dat_file in enumerate(dat_files):
        try:
            # Load and prepare airfoil
            x_coords, y_coords = parse_airfoil_file(dat_file)
            if x_coords is None or len(x_coords) != TARGET_POINTS:
                continue
            
            airfoil_name = dat_file.stem
            print(f"\n[{i+1}] Testing {airfoil_name}...")
            
            # Flatten coordinates
            original = np.concatenate([x_coords, y_coords])
            
            # Normalize
            original_normalized = scaler.transform(original.reshape(1, -1))
            original_tensor = torch.FloatTensor(original_normalized).to(DEVICE)
            
            # Encode
            with torch.no_grad():
                latent = encoder(original_tensor)
                latent_values = latent.cpu().numpy().flatten()
                print(f"  Latent shape: {latent.shape}")
                print(f"  Latent range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
                print(f"  Latent vector ({LATENT_DIM} values):")
                # Format as a readable array
                latent_str = ', '.join([f'{val:.6f}' for val in latent_values])
                print(f"    [{latent_str}]")
                
                # Decode
                reconstructed_normalized = decoder(latent)
            
            # Denormalize
            reconstructed = scaler.inverse_transform(reconstructed_normalized.cpu().numpy())
            reconstructed = reconstructed.flatten()
            
            # Calculate reconstruction error
            mse = np.mean((original - reconstructed) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(original - reconstructed))
            
            all_errors.append({
                'name': airfoil_name,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'latent': latent_values
            })
            
            print(f"  MSE: {mse:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  MAE: {mae:.6f}")
            
            # Split back into x and y
            x_reconstructed = reconstructed[:TARGET_POINTS]
            y_reconstructed = reconstructed[TARGET_POINTS:]
            
            # Plot comparison (optional, requires matplotlib)
            try:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.plot(x_coords, y_coords, 'b-', label='Original', linewidth=2)
                plt.plot(x_reconstructed, y_reconstructed, 'r--', label='Reconstructed', linewidth=2)
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'{airfoil_name} - Shape Comparison')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.axis('equal')
                
                plt.subplot(1, 2, 2)
                error_x = x_coords - x_reconstructed
                error_y = y_coords - y_reconstructed
                plt.plot(x_coords, error_x, 'g-', label='X error', linewidth=1.5)
                plt.plot(x_coords, error_y, 'm-', label='Y error', linewidth=1.5)
                plt.xlabel('x')
                plt.ylabel('Error')
                plt.title(f'{airfoil_name} - Reconstruction Error')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f'test_{airfoil_name}.png', dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved plot to test_{airfoil_name}.png")
            except Exception as e:
                print(f"  Could not create plot: {e}")
        
        except Exception as e:
            print(f"  Error processing {dat_file.name}: {e}")
            continue
    
    # Print summary statistics
    if all_errors:
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        avg_mse = np.mean([e['mse'] for e in all_errors])
        avg_rmse = np.mean([e['rmse'] for e in all_errors])
        avg_mae = np.mean([e['mae'] for e in all_errors])
        
        print(f"Average MSE:  {avg_mse:.6f}")
        print(f"Average RMSE: {avg_rmse:.6f}")
        print(f"Average MAE:  {avg_mae:.6f}")
        
        print(f"\nPer-airfoil errors and latent vectors:")
        for error in all_errors:
            print(f"  {error['name']:30s} - RMSE: {error['rmse']:.6f}, MAE: {error['mae']:.6f}")
            latent_str = ', '.join([f'{val:.6f}' for val in error['latent']])
            print(f"    Latent: [{latent_str}]")
    
    print(f"\n{'='*60}")
    print("✓ Autoencoder test completed!")

def test_latent_space():
    """Test operations in latent space (interpolation, etc.)."""
    if not (MODELS_DIR / "airfoil_encoder.pth").exists() or not (MODELS_DIR / "airfoil_decoder.pth").exists():
        print("Error: Model files not found. Please train the autoencoder first.")
        return
    
    encoder, decoder, _, _ = load_encoder_decoder()
    scaler = load_scaler()
    
    # Get two airfoils
    dat_files = sorted(BIGFOIL_DIR.glob("*.dat"))[:2]
    if len(dat_files) < 2:
        print("Need at least 2 airfoils for interpolation test")
        return
    
    print("\nTesting latent space interpolation...")
    
    airfoils = []
    latents = []
    
    for dat_file in dat_files:
        x_coords, y_coords = parse_airfoil_file(dat_file)
        if x_coords is None or len(x_coords) != TARGET_POINTS:
            continue
        
        original = np.concatenate([x_coords, y_coords])
        original_normalized = scaler.transform(original.reshape(1, -1))
        original_tensor = torch.FloatTensor(original_normalized).to(DEVICE)
        
        with torch.no_grad():
            latent = encoder(original_tensor)
        
        airfoils.append((x_coords, y_coords, dat_file.stem))
        latents.append(latent)
    
    if len(latents) < 2:
        print("Could not encode enough airfoils")
        return
    
    # Interpolate in latent space
    print(f"Interpolating between {airfoils[0][2]} and {airfoils[1][2]}...")
    
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    try:
        plt.figure(figsize=(15, 3))
        for idx, alpha in enumerate(alpha_values):
            # Linear interpolation in latent space
            interpolated_latent = (1 - alpha) * latents[0] + alpha * latents[1]
            
            # Decode
            with torch.no_grad():
                reconstructed_normalized = decoder(interpolated_latent)
            
            # Denormalize
            reconstructed = scaler.inverse_transform(reconstructed_normalized.cpu().numpy())
            reconstructed = reconstructed.flatten()
            
            x_interp = reconstructed[:TARGET_POINTS]
            y_interp = reconstructed[TARGET_POINTS:]
            
            plt.subplot(1, len(alpha_values), idx + 1)
            plt.plot(x_interp, y_interp, 'b-', linewidth=2)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'α = {alpha:.2f}')
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
        
        plt.tight_layout()
        plt.savefig('test_interpolation.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Saved interpolation plot to test_interpolation.png")
    except Exception as e:
        print(f"Could not create interpolation plot: {e}")

if __name__ == "__main__":
    print("="*60)
    print("Autoencoder Test Script")
    print("="*60)
    
    # Run basic tests
    test_autoencoder()
    
    # Test latent space operations
    print("\n")
    test_latent_space()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

