#!/usr/bin/env python3
"""
Generate CSV files with airfoil performance data using neuralFoil XXXL.
For each airfoil, generates data for angles from -10 to 15 degrees
and Reynolds numbers: 50k, 100k, 200k, 300k, 400k, 500k, 750k, 1m
Also includes latent vector encoding from the trained autoencoder.
"""

import os
import csv
import json
import numpy as np
from pathlib import Path
import neuralfoil as nf

# Optional imports for latent vector encoding
try:
    import torch
    import torch.nn as nn
    import pickle
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
BIGFOIL_DIR = PROJECT_ROOT / "data" / "bigfoil"
OUTPUT_DIR = PROJECT_ROOT / "data" / "airfoil_data"
MODELS_DIR = PROJECT_ROOT / "models"
ALPHA_RANGE = np.arange(-10, 15, 1)  # -10 to 15 degrees in 1 degree steps
REYNOLDS_NUMBERS = [50000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000]

# Autoencoder configuration for latent vectors
TARGET_POINTS = 200  # Number of points per airfoil for encoding
INPUT_DIM = 400  # 200 x-coordinates + 200 y-coordinates
if TORCH_AVAILABLE:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = None

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
        raise ValueError(f"Not enough coordinates in {filepath}")
    
    coords = np.array(coords)
    return coords[:, 0], coords[:, 1]

def resample_airfoil(x_coords, y_coords, target_points=200):
    """
    Resample airfoil coordinates to exactly target_points.
    Uses arc length parameterization for even spacing along the curve.
    """
    if len(x_coords) == target_points:
        return x_coords, y_coords
    
    # Compute arc length along the curve
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_normalized = s / s[-1]  # Normalize to [0, 1]
    
    # Create new parameterization with target_points
    s_new = np.linspace(0, 1, target_points)
    
    # Interpolate x and y coordinates
    x_new = np.interp(s_new, s_normalized, x_coords)
    y_new = np.interp(s_new, s_normalized, y_coords)
    
    return x_new, y_new

def load_encoder():
    """Load the saved encoder model. Returns None if not available."""
    if not TORCH_AVAILABLE:
        return None, None, None
    
    encoder_path = MODELS_DIR / "airfoil_encoder.pth"
    scaler_path = MODELS_DIR / "airfoil_scaler.pkl"
    
    if not encoder_path.exists() or not scaler_path.exists():
        return None, None, None
    
    try:
        # Load encoder
        encoder_checkpoint = torch.load(encoder_path, map_location=DEVICE)
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
        
        # Load scaler
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        
        print(f"✓ Encoder and scaler loaded successfully (latent dim: {encoder_checkpoint['latent_dim']})")
        return encoder, scaler, encoder_checkpoint['input_dim']
    except Exception as e:
        print(f"  Warning: Could not load encoder: {e}")
        return None, None, None

def encode_airfoil_coords(x_coords, y_coords, encoder, scaler, input_dim):
    """Encode airfoil coordinates to latent representation."""
    if encoder is None or scaler is None:
        return None
    
    try:
        # Resample to exactly TARGET_POINTS
        x_resampled, y_resampled = resample_airfoil(x_coords, y_coords, TARGET_POINTS)
        
        # Flatten: [x1, x2, ..., x200, y1, y2, ..., y200]
        flattened = np.concatenate([x_resampled, y_resampled])
        
        if len(flattened) != input_dim:
            return None
        
        # Normalize
        normalized = scaler.transform(flattened.reshape(1, -1))
        normalized_tensor = torch.FloatTensor(normalized).to(DEVICE)
        
        # Encode
        with torch.no_grad():
            latent = encoder(normalized_tensor)
            latent_values = latent.cpu().numpy().flatten()
        
        return latent_values
    
    except Exception as e:
        return None

def generate_airfoil_data(airfoil_name, x_coords, y_coords):
    """Generate performance data for an airfoil using neuralFoil XXXL.
    Returns a dictionary keyed by Reynolds number, with vectors for each."""
    data_by_re = {}
    
    # Create airfoil coordinates array (Nx2)
    coords = np.column_stack([x_coords, y_coords])
    
    for Re in REYNOLDS_NUMBERS:
        alpha_vector = []
        CL_vector = []
        CD_vector = []
        L_D_vector = []
        
        for alpha in ALPHA_RANGE:
            try:
                # Use neuralFoil XXXL model
                result = nf.get_aero_from_coordinates(
                    coordinates=coords,
                    alpha=alpha,
                    Re=Re,
                    model_size="xxxlarge"
                )
                
                # Extract CL and CD from result (they're numpy arrays, get scalar value)
                CL = float(result["CL"][0]) if isinstance(result["CL"], np.ndarray) else float(result["CL"])
                CD = float(result["CD"][0]) if isinstance(result["CD"], np.ndarray) else float(result["CD"])
                
                L_D = CL / CD if CD != 0 and not np.isnan(CD) else np.nan
                
                alpha_vector.append(alpha)
                CL_vector.append(CL)
                CD_vector.append(CD)
                L_D_vector.append(L_D)
            except Exception as e:
                print(f"  Error processing {airfoil_name} at alpha={alpha}, Re={Re}: {e}")
                # Continue with next angle
                continue
        
        data_by_re[Re] = {
            'alpha': alpha_vector,
            'CL': CL_vector,
            'CD': CD_vector,
            'L/D': L_D_vector
        }
    
    return data_by_re

def process_all_airfoils():
    """Process all airfoil files in the bigfoil directory."""
    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Try to load encoder (optional)
    encoder, scaler, input_dim = load_encoder()
    include_latent = encoder is not None and scaler is not None
    
    if include_latent:
        print(f"✓ Latent vectors will be included in CSV files")
    else:
        print(f"⚠ Encoder not available - CSV files will be generated without latent vectors")
        print(f"  (Train the autoencoder first to include latent vectors)")
    
    # Get all .dat files
    dat_files = sorted(BIGFOIL_DIR.glob("*.dat"))
    
    print(f"\nFound {len(dat_files)} airfoil files to process")
    
    for i, dat_file in enumerate(dat_files, 1):
        airfoil_name = dat_file.stem
        print(f"[{i}/{len(dat_files)}] Processing {airfoil_name}...")
        
        try:
            # Parse airfoil coordinates
            x_coords, y_coords = parse_airfoil_file(dat_file)
            
            # Encode to latent vector (if encoder is available)
            latent_vector = None
            if include_latent:
                latent_vector = encode_airfoil_coords(x_coords, y_coords, encoder, scaler, input_dim)
                if latent_vector is None:
                    print(f"  Warning: Could not encode {airfoil_name} to latent vector")
            
            # Generate performance data
            data_by_re = generate_airfoil_data(airfoil_name, x_coords, y_coords)
            
            if not data_by_re:
                print(f"  Warning: No data generated for {airfoil_name}")
                continue
            
            # Write CSV file: one row per Reynolds number with vectors
            csv_filename = OUTPUT_DIR / f"{airfoil_name}.csv"
            
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header (include latent_vector if available)
                header = ['Reynolds', 'alpha', 'Cl', 'cd', 'L/D']
                if include_latent and latent_vector is not None:
                    header.append('airfoil_latent_vector')
                writer.writerow(header)
                
                # Write one row per Reynolds number
                for Re in REYNOLDS_NUMBERS:
                    if Re in data_by_re:
                        data = data_by_re[Re]
                        # Convert vectors to JSON lists (replace NaN with None for JSON compatibility)
                        L_D_list = [None if np.isnan(ld) else ld for ld in data['L/D']]
                        alpha_json = json.dumps(data['alpha'])
                        CL_json = json.dumps(data['CL'])
                        CD_json = json.dumps(data['CD'])
                        L_D_json = json.dumps(L_D_list)
                        
                        row = [Re, alpha_json, CL_json, CD_json, L_D_json]
                        
                        # Add latent vector as JSON list if available
                        if include_latent and latent_vector is not None:
                            latent_json = json.dumps(latent_vector.tolist())
                            row.append(latent_json)
                        
                        writer.writerow(row)
            
            total_points = sum(len(data['alpha']) for data in data_by_re.values())
            status_msg = f"  ✓ Generated {csv_filename} with {len(data_by_re)} Reynolds numbers, {total_points} total data points"
            if include_latent and latent_vector is not None:
                status_msg += " (with latent vector)"
            print(status_msg)
            
        except Exception as e:
            print(f"  ✗ Error processing {airfoil_name}: {e}")
            continue
    
    print(f"\nCompleted! CSV files saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    process_all_airfoils()

