#!/usr/bin/env python3
"""
Create a CSV file containing both xfoil_latent_vector and airfoil_latent_vector for each airfoil.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import csv
import json

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
AIRFOIL_DATA_DIR = PROJECT_ROOT / "data" / "airfoil_data"
BIGFOIL_DIR = PROJECT_ROOT / "data" / "bigfoil"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_CSV = PROJECT_ROOT / "data" / "latent_vectors.csv"
INPUT_DIM_XFOIL = 101  # 1 (Reynolds) + 25 (alpha) + 25 (Cl) + 25 (cd) + 25 (L/D)
ALPHA_LENGTH = 25
TARGET_POINTS = 200  # For airfoil coordinates
INPUT_DIM_AIRFOIL = 400  # 200 x + 200 y
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_xfoil_encoder():
    """Load the xfoil (performance) encoder."""
    encoder_path = MODELS_DIR / "xfoil_encoder.pth"
    scaler_path = MODELS_DIR / "xfoil_scaler.pkl"
    
    encoder = None
    scaler = None
    input_dim = None
    
    if encoder_path.exists():
        encoder_checkpoint = torch.load(encoder_path, map_location=DEVICE)
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
        input_dim = encoder_checkpoint['input_dim']
        print(f"✓ Loaded xfoil encoder from {encoder_path}")
    
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✓ Loaded xfoil scaler from {scaler_path}")
    
    return encoder, scaler, input_dim

def load_airfoil_encoder():
    """Load the airfoil (coordinate) encoder."""
    encoder_path = MODELS_DIR / "airfoil_encoder.pth"
    scaler_path = MODELS_DIR / "airfoil_scaler.pkl"
    
    encoder = None
    scaler = None
    input_dim = None
    
    if encoder_path.exists():
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
        input_dim = encoder_checkpoint['input_dim']
        print(f"✓ Loaded airfoil encoder from {encoder_path}")
    
    if scaler_path.exists():
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        print(f"✓ Loaded airfoil scaler from {scaler_path}")
    
    return encoder, scaler, input_dim

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

def resample_airfoil(x_coords, y_coords, target_points=200):
    """Resample airfoil coordinates to exactly target_points."""
    if len(x_coords) == target_points:
        return x_coords, y_coords
    
    # Compute arc length along the curve
    dx = np.diff(x_coords)
    dy = np.diff(y_coords)
    ds = np.sqrt(dx**2 + dy**2)
    s = np.concatenate([[0], np.cumsum(ds)])
    s_normalized = s / s[-1] if s[-1] > 0 else s
    
    # Create new parameterization
    s_new = np.linspace(0, 1, target_points)
    
    # Interpolate
    x_new = np.interp(s_new, s_normalized, x_coords)
    y_new = np.interp(s_new, s_normalized, y_coords)
    
    return x_new, y_new

def encode_airfoil_coords(x_coords, y_coords, encoder, scaler, input_dim):
    """Encode airfoil coordinates to latent representation."""
    try:
        x_resampled, y_resampled = resample_airfoil(x_coords, y_coords, TARGET_POINTS)
        flattened = np.concatenate([x_resampled, y_resampled])
        
        if len(flattened) != input_dim:
            return None
        
        normalized = scaler.transform(flattened.reshape(1, -1))
        normalized_tensor = torch.FloatTensor(normalized).to(DEVICE)
        
        with torch.no_grad():
            latent = encoder(normalized_tensor)
            return latent.cpu().numpy().flatten()
    except Exception as e:
        return None

def parse_performance_row(row):
    """Parse a CSV row into performance data."""
    reynolds = float(row[0])
    alpha = json.loads(row[1])
    cl = json.loads(row[2])
    cd = json.loads(row[3])
    l_d = json.loads(row[4])
    
    # Handle None values in L/D
    l_d = [0.0 if x is None else float(x) for x in l_d]
    
    # Normalize Reynolds (log scale)
    reynolds_log = np.log10(reynolds)
    
    # Flatten
    flattened = np.array([reynolds_log] + alpha + cl + cd + l_d)
    
    return flattened

def encode_performance_data(flattened, encoder, scaler):
    """Encode performance data to xfoil latent representation."""
    try:
        if len(flattened) != INPUT_DIM_XFOIL:
            return None
        
        normalized = scaler.transform(flattened.reshape(1, -1))
        normalized_tensor = torch.FloatTensor(normalized).to(DEVICE)
        
        with torch.no_grad():
            latent = encoder(normalized_tensor)
            return latent.cpu().numpy().flatten()
    except Exception as e:
        return None

def main():
    """Main function to create the latent vectors CSV."""
    # Load encoders
    print("Loading encoders...")
    xfoil_encoder, xfoil_scaler, xfoil_input_dim = load_xfoil_encoder()
    airfoil_encoder, airfoil_scaler, airfoil_input_dim = load_airfoil_encoder()
    
    if xfoil_encoder is None or xfoil_scaler is None:
        print("Error: Could not load xfoil encoder/scaler")
        return
    
    if airfoil_encoder is None or airfoil_scaler is None:
        print("Error: Could not load airfoil encoder/scaler")
        return
    
    # Get all CSV files
    csv_files = sorted(AIRFOIL_DATA_DIR.glob("*.csv"))
    print(f"\nFound {len(csv_files)} CSV files to process")
    
    results = []
    
    for i, csv_file in enumerate(csv_files, 1):
        if i % 1000 == 0:
            print(f"Processing {i}/{len(csv_files)}...")
        
        airfoil_name = csv_file.stem
        
        try:
            # Read CSV to get performance data and airfoil_latent_vector
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                
                # Find column indices
                try:
                    alpha_idx = header.index('alpha')
                    cl_idx = header.index('Cl')
                    cd_idx = header.index('cd')
                    l_d_idx = header.index('L/D')
                    airfoil_latent_idx = header.index('airfoil_latent_vector')
                except ValueError:
                    continue
                
                # Get airfoil_latent_vector once (same for all Reynolds numbers)
                # Read first row to get airfoil_latent_vector
                first_row = next(reader)
                try:
                    airfoil_latent_str = first_row[airfoil_latent_idx]
                    airfoil_latent = json.loads(airfoil_latent_str)
                    airfoil_latent = [float(x) for x in airfoil_latent]
                except (json.JSONDecodeError, IndexError, ValueError):
                    # If not available, encode from coordinates
                    dat_file = BIGFOIL_DIR / f"{airfoil_name}.dat"
                    if dat_file.exists():
                        x_coords, y_coords = parse_airfoil_file(dat_file)
                        if x_coords is not None:
                            airfoil_latent = encode_airfoil_coords(
                                x_coords, y_coords, airfoil_encoder, airfoil_scaler, airfoil_input_dim
                            )
                            if airfoil_latent is None:
                                continue
                            airfoil_latent = airfoil_latent.tolist()
                        else:
                            continue
                    else:
                        continue
                
                # Process all rows (all Reynolds numbers)
                # Process first row
                try:
                    performance_data = parse_performance_row([
                        first_row[0], first_row[alpha_idx], first_row[cl_idx], first_row[cd_idx], first_row[l_d_idx]
                    ])
                    xfoil_latent = encode_performance_data(
                        performance_data, xfoil_encoder, xfoil_scaler
                    )
                    if xfoil_latent is not None:
                        results.append({
                            'airfoil_name': airfoil_name,
                            'Reynolds': int(float(first_row[0])),
                            'xfoil_latent_vector': xfoil_latent.tolist(),
                            'airfoil_latent_vector': airfoil_latent
                        })
                except (IndexError, ValueError, json.JSONDecodeError):
                    pass
                
                # Process remaining rows
                for row in reader:
                    try:
                        performance_data = parse_performance_row([
                            row[0], row[alpha_idx], row[cl_idx], row[cd_idx], row[l_d_idx]
                        ])
                        xfoil_latent = encode_performance_data(
                            performance_data, xfoil_encoder, xfoil_scaler
                        )
                        if xfoil_latent is not None:
                            results.append({
                                'airfoil_name': airfoil_name,
                                'Reynolds': int(float(row[0])),
                                'xfoil_latent_vector': xfoil_latent.tolist(),
                                'airfoil_latent_vector': airfoil_latent
                            })
                    except (IndexError, ValueError, json.JSONDecodeError):
                        continue
        
        except Exception as e:
            continue
    
    # Write results to CSV
    print(f"\nWriting {len(results)} results to {OUTPUT_CSV}...")
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['airfoil_name', 'Reynolds', 'xfoil_latent_vector', 'airfoil_latent_vector'])
        
        for result in results:
            xfoil_json = json.dumps(result['xfoil_latent_vector'])
            airfoil_json = json.dumps(result['airfoil_latent_vector'])
            writer.writerow([result['airfoil_name'], result['Reynolds'], xfoil_json, airfoil_json])
    
    print(f"✓ Successfully created {OUTPUT_CSV} with {len(results)} rows ({len(results) // 8} airfoils, ~8 Reynolds numbers each)")

if __name__ == "__main__":
    main()

