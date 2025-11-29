#!/usr/bin/env python3
"""
Test script for the trained performance autoencoder.
Tests encoding and decoding of airfoil performance data (Reynolds, alpha, Cl, cd, L/D).
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import csv
import json
import matplotlib.pyplot as plt

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
AIRFOIL_DATA_DIR = PROJECT_ROOT / "data" / "airfoil_data"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 16
INPUT_DIM = 101  # 1 (Reynolds) + 25 (alpha) + 25 (Cl) + 25 (cd) + 25 (L/D)
ALPHA_LENGTH = 25
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_encoder_decoder():
    """Load the saved encoder and decoder models."""
    # Load encoder
    encoder_checkpoint = torch.load(MODELS_DIR / "xfoil_encoder.pth", map_location=DEVICE)
    encoder = nn.Sequential(
        nn.Linear(encoder_checkpoint['input_dim'], 128),
        nn.Tanh(),
        nn.Linear(128, 64),
        nn.Tanh(),
        nn.Linear(64, 32),
        nn.Tanh(),
        nn.Linear(32, 16),
        nn.Tanh(),
        nn.Linear(16, encoder_checkpoint['latent_dim'])
    ).to(DEVICE)
    encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    encoder.eval()
    
    # Load decoder
    decoder_checkpoint = torch.load(MODELS_DIR / "xfoil_decoder.pth", map_location=DEVICE)
    decoder = nn.Sequential(
        nn.Linear(decoder_checkpoint['input_dim'], 16),
        nn.Tanh(),
        nn.Linear(16, 32),
        nn.Tanh(),
        nn.Linear(32, 64),
        nn.Tanh(),
        nn.Linear(64, 128),
        nn.Tanh(),
        nn.Linear(128, decoder_checkpoint['output_dim'])
    ).to(DEVICE)
    decoder.load_state_dict(decoder_checkpoint['model_state_dict'])
    decoder.eval()
    
    print("✓ Encoder and decoder loaded successfully")
    return encoder, decoder, encoder_checkpoint['input_dim'], decoder_checkpoint['output_dim']

def load_scaler():
    """Load the data scaler."""
    with open(MODELS_DIR / "xfoil_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print("✓ Scaler loaded successfully")
    return scaler

def parse_performance_row(row):
    """Parse a CSV row into performance data."""
    reynolds = float(row[0])
    alpha = json.loads(row[1])
    cl = json.loads(row[2])
    cd = json.loads(row[3])
    l_d = json.loads(row[4])
    
    # Handle None values in L/D
    l_d = [0.0 if x is None else float(x) for x in l_d]
    
    # Normalize Reynolds
    reynolds_log = np.log10(reynolds)
    
    # Flatten
    flattened = np.array([reynolds_log] + alpha + cl + cd + l_d)
    
    return flattened, reynolds, alpha, cl, cd, l_d

def test_autoencoder():
    """Test the autoencoder on sample performance data."""
    # Check if models exist
    if not (MODELS_DIR / "xfoil_encoder.pth").exists() or not (MODELS_DIR / "xfoil_decoder.pth").exists():
        print("Error: Model files not found. Please train the autoencoder first.")
        return
    
    if not (MODELS_DIR / "xfoil_scaler.pkl").exists():
        print("Error: Scaler file not found. Please train the autoencoder first.")
        return
    
    # Load models and scaler
    encoder, decoder, input_dim, output_dim = load_encoder_decoder()
    scaler = load_scaler()
    
    print(f"\nModel architecture:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Latent dimension: {LATENT_DIM}")
    print(f"  Output dimension: {output_dim}")
    
    # Get some test CSV files
    csv_files = sorted(AIRFOIL_DATA_DIR.glob("*.csv"))[:5]  # Test on first 5 airfoils
    print(f"\nTesting on {len(csv_files)} airfoils...")
    
    all_errors = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            airfoil_name = csv_file.stem
            print(f"\n[{i+1}] Testing {airfoil_name}...")
            
            # Read first row (first Reynolds number) from CSV
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                if len(header) < 5:
                    continue
                
                row = next(reader)  # Get first data row
                
                # Parse original data
                original_flattened, reynolds, alpha, cl, cd, l_d = parse_performance_row(row)
                
                if len(original_flattened) != INPUT_DIM:
                    print(f"  Warning: Wrong dimensions: {len(original_flattened)} != {INPUT_DIM}")
                    continue
                
                # Normalize
                original_normalized = scaler.transform(original_flattened.reshape(1, -1))
                original_tensor = torch.FloatTensor(original_normalized).to(DEVICE)
                
                # Encode
                with torch.no_grad():
                    latent = encoder(original_tensor)
                    latent_values = latent.cpu().numpy().flatten()
                    print(f"  Latent shape: {latent.shape}")
                    print(f"  Latent range: [{latent.min().item():.4f}, {latent.max().item():.4f}]")
                    print(f"  Latent vector ({LATENT_DIM} values):")
                    latent_str = ', '.join([f'{val:.6f}' for val in latent_values])
                    print(f"    [{latent_str}]")
                    
                    # Decode
                    reconstructed_normalized = decoder(latent)
                
                # Denormalize
                reconstructed = scaler.inverse_transform(reconstructed_normalized.cpu().numpy())
                reconstructed = reconstructed.flatten()
                
                # Calculate reconstruction error
                mse = np.mean((original_flattened - reconstructed) ** 2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(original_flattened - reconstructed))
                
                all_errors.append({
                    'name': airfoil_name,
                    'reynolds': reynolds,
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'latent': latent_values
                })
                
                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  MAE: {mae:.6f}")
                
                # Extract components from reconstructed data
                reynolds_recon = 10 ** reconstructed[0]
                alpha_recon = reconstructed[1:1+ALPHA_LENGTH].tolist()
                cl_recon = reconstructed[1+ALPHA_LENGTH:1+2*ALPHA_LENGTH].tolist()
                cd_recon = reconstructed[1+2*ALPHA_LENGTH:1+3*ALPHA_LENGTH].tolist()
                l_d_recon = reconstructed[1+3*ALPHA_LENGTH:1+4*ALPHA_LENGTH].tolist()
                
                # Plot comparison
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    
                    # Cl vs alpha
                    axes[0, 0].plot(alpha, cl, 'b-', label='Original', linewidth=2)
                    axes[0, 0].plot(alpha_recon, cl_recon, 'r--', label='Reconstructed', linewidth=2)
                    axes[0, 0].set_xlabel('Alpha (degrees)')
                    axes[0, 0].set_ylabel('Cl')
                    axes[0, 0].set_title(f'{airfoil_name} - Cl vs Alpha (Re={reynolds:.0f})')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # cd vs alpha
                    axes[0, 1].plot(alpha, cd, 'b-', label='Original', linewidth=2)
                    axes[0, 1].plot(alpha_recon, cd_recon, 'r--', label='Reconstructed', linewidth=2)
                    axes[0, 1].set_xlabel('Alpha (degrees)')
                    axes[0, 1].set_ylabel('cd')
                    axes[0, 1].set_title(f'cd vs Alpha')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # L/D vs alpha
                    axes[1, 0].plot(alpha, l_d, 'b-', label='Original', linewidth=2)
                    axes[1, 0].plot(alpha_recon, l_d_recon, 'r--', label='Reconstructed', linewidth=2)
                    axes[1, 0].set_xlabel('Alpha (degrees)')
                    axes[1, 0].set_ylabel('L/D')
                    axes[1, 0].set_title('L/D vs Alpha')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Error plot
                    cl_error = np.array(cl) - np.array(cl_recon)
                    cd_error = np.array(cd) - np.array(cd_recon)
                    axes[1, 1].plot(alpha, cl_error, 'g-', label='Cl error', linewidth=1.5)
                    axes[1, 1].plot(alpha, cd_error, 'm-', label='cd error', linewidth=1.5)
                    axes[1, 1].set_xlabel('Alpha (degrees)')
                    axes[1, 1].set_ylabel('Error')
                    axes[1, 1].set_title('Reconstruction Error')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'test_performance_{airfoil_name}_Re{reynolds:.0f}.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"  Saved plot to test_performance_{airfoil_name}_Re{reynolds:.0f}.png")
                    
                    # Print some statistics
                    print(f"  Reynolds: {reynolds:.0f} -> {reynolds_recon:.0f} (error: {abs(reynolds - reynolds_recon):.0f})")
                    
                except Exception as e:
                    print(f"  Could not create plot: {e}")
        
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
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
            print(f"  {error['name']:30s} - Re: {error['reynolds']:7.0f} - RMSE: {error['rmse']:.6f}, MAE: {error['mae']:.6f}")
            latent_str = ', '.join([f'{val:.6f}' for val in error['latent']])
            print(f"    Latent: [{latent_str}]")
    
    print(f"\n{'='*60}")
    print("✓ Performance autoencoder test completed!")

if __name__ == "__main__":
    print("="*60)
    print("Performance Autoencoder Test Script")
    print("="*60)
    
    test_autoencoder()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

