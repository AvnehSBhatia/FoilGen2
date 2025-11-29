#!/usr/bin/env python3
"""
Full pipeline test: xfoil data -> encode -> latent mapper -> decode -> compare with original .dat file.
Tests the complete pipeline from performance data to airfoil coordinates.
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import pickle
import csv
import json
import matplotlib.pyplot as plt
import neuralfoil as nf

# Configuration - paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
AIRFOIL_DATA_DIR = PROJECT_ROOT / "data" / "airfoil_data"
BIGFOIL_DIR = PROJECT_ROOT / "data" / "bigfoil"
MODELS_DIR = PROJECT_ROOT / "models"
LATENT_DIM = 16
INPUT_DIM_XFOIL = 101  # 1 (Reynolds) + 25 (alpha) + 25 (Cl) + 25 (cd) + 25 (L/D)
ALPHA_LENGTH = 25
ALPHA_RANGE = np.arange(-10, 15, 1)  # -10 to 15 degrees in 1 degree steps
TARGET_POINTS = 200  # Number of points per airfoil
INPUT_DIM_AIRFOIL = 400  # 200 x-coordinates + 200 y-coordinates
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

def load_xfoil_encoder():
    """Load the xfoil (performance) encoder and scaler."""
    encoder_path = MODELS_DIR / "xfoil_encoder.pth"
    scaler_path = MODELS_DIR / "xfoil_scaler.pkl"
    
    if not encoder_path.exists() or not scaler_path.exists():
        return None, None, None
    
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
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Loaded xfoil encoder and scaler")
    return encoder, scaler, encoder_checkpoint['input_dim']

class LatentMapper(nn.Module):
    """Neural network to map xfoil latent vector to airfoil latent vector."""
    def __init__(self, input_dim=16, output_dim=16, hidden_dims=[128,256,128]):
        super(LatentMapper, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x): 
        return self.network(x)

def load_latent_mapper():
    """Load the latent mapper model and scalers."""
    mapper_path = MODELS_DIR / "latent_mapper.pth"
    xfoil_scaler_path = MODELS_DIR / "latent_mapper_xfoil_scaler.pkl"
    airfoil_scaler_path = MODELS_DIR / "latent_mapper_airfoil_scaler.pkl"
    
    if not mapper_path.exists() or not xfoil_scaler_path.exists() or not airfoil_scaler_path.exists():
        return None, None, None
    
    mapper_checkpoint = torch.load(mapper_path, map_location=DEVICE)
    mapper = LatentMapper(input_dim=LATENT_DIM, output_dim=LATENT_DIM).to(DEVICE)
    mapper.load_state_dict(mapper_checkpoint['model_state_dict'])
    mapper.eval()
    
    with open(xfoil_scaler_path, "rb") as f:
        xfoil_scaler = pickle.load(f)
    
    with open(airfoil_scaler_path, "rb") as f:
        airfoil_scaler = pickle.load(f)
    
    print("✓ Loaded latent mapper and scalers")
    return mapper, xfoil_scaler, airfoil_scaler

def load_airfoil_decoder():
    """Load the airfoil (coordinate) decoder and scaler."""
    decoder_path = MODELS_DIR / "airfoil_decoder.pth"
    scaler_path = MODELS_DIR / "airfoil_scaler.pkl"
    
    if not decoder_path.exists() or not scaler_path.exists():
        return None, None, None
    
    decoder_checkpoint = torch.load(decoder_path, map_location=DEVICE)
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
    
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    
    print("✓ Loaded airfoil decoder and scaler")
    return decoder, scaler, decoder_checkpoint['output_dim']

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
    """
    Resample airfoil to target number of points using arc-length parameterization.
    """
    # Calculate arc length
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
    
    return flattened, reynolds, alpha, cl, cd, l_d

def evaluate_airfoil_with_neuralfoil(x_coords, y_coords, reynolds, alpha_range):
    """Evaluate airfoil performance using neuralfoil XXXL."""
    # Create airfoil coordinates array (Nx2)
    coords = np.column_stack([x_coords, y_coords])
    
    alpha_vector = []
    CL_vector = []
    CD_vector = []
    L_D_vector = []
    
    for alpha in alpha_range:
        try:
            # Use neuralFoil XXXL model
            result = nf.get_aero_from_coordinates(
                coordinates=coords,
                alpha=alpha,
                Re=reynolds,
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
            print(f"    Warning: Error at alpha={alpha}, Re={reynolds}: {e}")
            # Use NaN for failed points
            alpha_vector.append(alpha)
            CL_vector.append(np.nan)
            CD_vector.append(np.nan)
            L_D_vector.append(np.nan)
    
    return {
        'alpha': np.array(alpha_vector),
        'CL': np.array(CL_vector),
        'CD': np.array(CD_vector),
        'L/D': np.array(L_D_vector)
    }

def test_full_pipeline():
    """Test the full pipeline: xfoil data -> encode -> latent mapper -> decode -> compare."""
    # Check if all models exist
    required_files = [
        "xfoil_encoder.pth",
        "xfoil_scaler.pkl",
        "latent_mapper.pth",
        "latent_mapper_xfoil_scaler.pkl",
        "latent_mapper_airfoil_scaler.pkl",
        "airfoil_decoder.pth",
        "airfoil_scaler.pkl"
    ]
    
    missing_files = [f for f in required_files if not (MODELS_DIR / f).exists()]
    if missing_files:
        print(f"Error: Missing required model files: {missing_files}")
        print("Please train all models first.")
        return
    
    # Load all models
    print("\nLoading models...")
    xfoil_encoder, xfoil_scaler, xfoil_input_dim = load_xfoil_encoder()
    latent_mapper, mapper_xfoil_scaler, mapper_airfoil_scaler = load_latent_mapper()
    airfoil_decoder, airfoil_scaler, airfoil_output_dim = load_airfoil_decoder()
    
    if xfoil_encoder is None or latent_mapper is None or airfoil_decoder is None:
        print("Error: Could not load all required models")
        return
    
    print(f"\nPipeline architecture:")
    print(f"  XFoil input: {xfoil_input_dim} dimensions")
    print(f"  XFoil latent: {LATENT_DIM} dimensions")
    print(f"  Airfoil latent: {LATENT_DIM} dimensions")
    print(f"  Airfoil output: {airfoil_output_dim} dimensions")
    
    # Get test files
    csv_files = sorted(AIRFOIL_DATA_DIR.glob("*.csv"))[0:15]  # Test on first 10 airfoils
    print(f"\nTesting on {len(csv_files)} airfoils...")
    
    all_results = []
    
    for i, csv_file in enumerate(csv_files):
        try:
            airfoil_name = csv_file.stem
            print(f"\n[{i+1}] Testing {airfoil_name}...")
            
            # Check if corresponding .dat file exists
            dat_file = BIGFOIL_DIR / f"{airfoil_name}.dat"
            if not dat_file.exists():
                print(f"  Warning: No corresponding .dat file found: {dat_file}")
                continue
            
            # Load original airfoil coordinates
            x_original, y_original = parse_airfoil_file(dat_file)
            if x_original is None:
                print(f"  Warning: Could not parse .dat file")
                continue
            
            # Resample original to target points
            x_original_resampled, y_original_resampled = resample_airfoil(
                x_original, y_original, target_points=TARGET_POINTS
            )
            
            # Read CSV and find row with 500k Reynolds number
            target_reynolds = 500000
            row = None
            reynolds = None
            
            with open(csv_file, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                if len(header) < 5:
                    continue
                
                # Search for row with target Reynolds number
                for csv_row in reader:
                    try:
                        row_reynolds = float(csv_row[0])
                        if abs(row_reynolds - target_reynolds) < 1:  # Allow small floating point differences
                            row = csv_row
                            reynolds = row_reynolds
                            break
                    except (ValueError, IndexError):
                        continue
            
            if row is None:
                print(f"  Warning: No row found with Re={target_reynolds}")
                continue
            
            # Parse performance data
            performance_data, reynolds, alpha_original, cl_original, cd_original, l_d_original = parse_performance_row(row)
            
            if len(performance_data) != INPUT_DIM_XFOIL:
                print(f"  Warning: Wrong dimensions: {len(performance_data)} != {INPUT_DIM_XFOIL}")
                continue
            
            # Step 1: Encode xfoil performance data
            performance_normalized = xfoil_scaler.transform(performance_data.reshape(1, -1))
            performance_tensor = torch.FloatTensor(performance_normalized).to(DEVICE)
            
            with torch.no_grad():
                xfoil_latent = xfoil_encoder(performance_tensor)
                xfoil_latent_np = xfoil_latent.cpu().numpy().flatten()
            
            print(f"  XFoil latent: [{', '.join([f'{v:.4f}' for v in xfoil_latent_np])}]")
            
            # Step 2: Map through latent mapper
            xfoil_latent_normalized = mapper_xfoil_scaler.transform(xfoil_latent_np.reshape(1, -1))
            xfoil_latent_tensor = torch.FloatTensor(xfoil_latent_normalized).to(DEVICE)
            
            with torch.no_grad():
                airfoil_latent_normalized = latent_mapper(xfoil_latent_tensor)
                airfoil_latent_normalized_np = airfoil_latent_normalized.cpu().numpy().flatten()
            
            # Denormalize airfoil latent
            airfoil_latent = mapper_airfoil_scaler.inverse_transform(airfoil_latent_normalized_np.reshape(1, -1))
            airfoil_latent_np = airfoil_latent.flatten()
            
            print(f"  Airfoil latent: [{', '.join([f'{v:.4f}' for v in airfoil_latent_np])}]")
            
            # Step 3: Decode to airfoil coordinates
            airfoil_latent_tensor = torch.FloatTensor(airfoil_latent_np).to(DEVICE)
            
            with torch.no_grad():
                reconstructed_normalized = airfoil_decoder(airfoil_latent_tensor.unsqueeze(0))
            
            # Denormalize
            reconstructed = airfoil_scaler.inverse_transform(reconstructed_normalized.cpu().numpy())
            reconstructed = reconstructed.flatten()
            
            # Split into x and y coordinates
            x_reconstructed = reconstructed[:TARGET_POINTS]
            y_reconstructed = reconstructed[TARGET_POINTS:]
            
            # Evaluate reconstructed airfoil with neuralfoil
            print(f"  Evaluating reconstructed airfoil with neuralfoil XXXL...")
            neuralfoil_results = evaluate_airfoil_with_neuralfoil(
                x_reconstructed, y_reconstructed, reynolds, ALPHA_RANGE
            )
            
            # Compare with original performance data
            cl_reconstructed = neuralfoil_results['CL']
            cd_reconstructed = neuralfoil_results['CD']
            l_d_reconstructed = neuralfoil_results['L/D']
            alpha_reconstructed = neuralfoil_results['alpha']
            
            # Convert original to numpy arrays for comparison
            cl_original_arr = np.array(cl_original)
            cd_original_arr = np.array(cd_original)
            l_d_original_arr = np.array(l_d_original)
            alpha_original_arr = np.array(alpha_original)
            
            # Calculate performance comparison metrics
            # Filter out NaN values for comparison
            valid_mask = ~(np.isnan(cl_reconstructed) | np.isnan(cl_original_arr))
            if np.any(valid_mask):
                cl_rmse = np.sqrt(np.mean((cl_original_arr[valid_mask] - cl_reconstructed[valid_mask]) ** 2))
                cl_mae = np.mean(np.abs(cl_original_arr[valid_mask] - cl_reconstructed[valid_mask]))
                
                valid_mask_cd = ~(np.isnan(cd_reconstructed) | np.isnan(cd_original_arr))
                cd_rmse = np.sqrt(np.mean((cd_original_arr[valid_mask_cd] - cd_reconstructed[valid_mask_cd]) ** 2)) if np.any(valid_mask_cd) else np.nan
                cd_mae = np.mean(np.abs(cd_original_arr[valid_mask_cd] - cd_reconstructed[valid_mask_cd])) if np.any(valid_mask_cd) else np.nan
                
                valid_mask_ld = ~(np.isnan(l_d_reconstructed) | np.isnan(l_d_original_arr))
                ld_rmse = np.sqrt(np.mean((l_d_original_arr[valid_mask_ld] - l_d_reconstructed[valid_mask_ld]) ** 2)) if np.any(valid_mask_ld) else np.nan
                ld_mae = np.mean(np.abs(l_d_original_arr[valid_mask_ld] - l_d_reconstructed[valid_mask_ld])) if np.any(valid_mask_ld) else np.nan
            else:
                cl_rmse = cl_mae = cd_rmse = cd_mae = ld_rmse = ld_mae = np.nan
            
            print(f"  Performance comparison:")
            print(f"    Cl - RMSE: {cl_rmse:.6f}, MAE: {cl_mae:.6f}")
            print(f"    cd - RMSE: {cd_rmse:.6f}, MAE: {cd_mae:.6f}")
            print(f"    L/D - RMSE: {ld_rmse:.6f}, MAE: {ld_mae:.6f}")
            
            # Calculate comparison metrics
            # For x coordinates (should be similar)
            x_mse = np.mean((x_original_resampled - x_reconstructed) ** 2)
            x_rmse = np.sqrt(x_mse)
            x_mae = np.mean(np.abs(x_original_resampled - x_reconstructed))
            
            # For y coordinates (main comparison)
            y_mse = np.mean((y_original_resampled - y_reconstructed) ** 2)
            y_rmse = np.sqrt(y_mse)
            y_mae = np.mean(np.abs(y_original_resampled - y_reconstructed))
            
            # Overall metrics
            overall_mse = np.mean((np.concatenate([x_original_resampled, y_original_resampled]) - 
                                  np.concatenate([x_reconstructed, y_reconstructed])) ** 2)
            overall_rmse = np.sqrt(overall_mse)
            overall_mae = np.mean(np.abs(np.concatenate([x_original_resampled, y_original_resampled]) - 
                                         np.concatenate([x_reconstructed, y_reconstructed])))
            
            all_results.append({
                'name': airfoil_name,
                'reynolds': reynolds,
                'x_rmse': x_rmse,
                'x_mae': x_mae,
                'y_rmse': y_rmse,
                'y_mae': y_mae,
                'overall_rmse': overall_rmse,
                'overall_mae': overall_mae,
                'cl_rmse': cl_rmse,
                'cl_mae': cl_mae,
                'cd_rmse': cd_rmse,
                'cd_mae': cd_mae,
                'ld_rmse': ld_rmse,
                'ld_mae': ld_mae,
                'x_reconstructed': x_reconstructed,
                'y_reconstructed': y_reconstructed,
                'alpha_original': alpha_original_arr,
                'cl_original': cl_original_arr,
                'cd_original': cd_original_arr,
                'l_d_original': l_d_original_arr,
                'alpha_reconstructed': alpha_reconstructed,
                'cl_reconstructed': cl_reconstructed,
                'cd_reconstructed': cd_reconstructed,
                'l_d_reconstructed': l_d_reconstructed
            })
            
            print(f"  X coordinates - RMSE: {x_rmse:.6f}, MAE: {x_mae:.6f}")
            print(f"  Y coordinates - RMSE: {y_rmse:.6f}, MAE: {y_mae:.6f}")
            print(f"  Overall - RMSE: {overall_rmse:.6f}, MAE: {overall_mae:.6f}")
            
            # Plot comparison
            try:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                
                # Original vs reconstructed airfoil
                axes[0, 0].plot(x_original_resampled, y_original_resampled, 'b-', 
                           label='Original', linewidth=2, alpha=0.7)
                axes[0, 0].plot(x_reconstructed, y_reconstructed, 'r--', 
                           label='Reconstructed', linewidth=2, alpha=0.7)
                axes[0, 0].set_xlabel('X')
                axes[0, 0].set_ylabel('Y')
                axes[0, 0].set_title(f'{airfoil_name} - Airfoil Comparison (Re={reynolds:.0f})')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                axes[0, 0].axis('equal')
                
                # Error plot
                y_error = y_original_resampled - y_reconstructed
                axes[0, 1].plot(x_original_resampled, y_error, 'g-', linewidth=1.5)
                axes[0, 1].set_xlabel('X')
                axes[0, 1].set_ylabel('Y Error')
                axes[0, 1].set_title(f'Y Coordinate Error (RMSE: {y_rmse:.6f})')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                # Cl vs alpha comparison
                axes[0, 2].plot(alpha_original_arr, cl_original_arr, 'b-', 
                              label='Original', linewidth=2, alpha=0.7)
                axes[0, 2].plot(alpha_reconstructed, cl_reconstructed, 'r--', 
                              label='Reconstructed', linewidth=2, alpha=0.7)
                axes[0, 2].set_xlabel('Alpha (degrees)')
                axes[0, 2].set_ylabel('Cl')
                axes[0, 2].set_title(f'Cl vs Alpha (RMSE: {cl_rmse:.6f})')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
                
                # cd vs alpha comparison
                axes[1, 0].plot(alpha_original_arr, cd_original_arr, 'b-', 
                              label='Original', linewidth=2, alpha=0.7)
                axes[1, 0].plot(alpha_reconstructed, cd_reconstructed, 'r--', 
                              label='Reconstructed', linewidth=2, alpha=0.7)
                axes[1, 0].set_xlabel('Alpha (degrees)')
                axes[1, 0].set_ylabel('cd')
                axes[1, 0].set_title(f'cd vs Alpha (RMSE: {cd_rmse:.6f})')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # L/D vs alpha comparison
                axes[1, 1].plot(alpha_original_arr, l_d_original_arr, 'b-', 
                              label='Original', linewidth=2, alpha=0.7)
                axes[1, 1].plot(alpha_reconstructed, l_d_reconstructed, 'r--', 
                              label='Reconstructed', linewidth=2, alpha=0.7)
                axes[1, 1].set_xlabel('Alpha (degrees)')
                axes[1, 1].set_ylabel('L/D')
                axes[1, 1].set_title(f'L/D vs Alpha (RMSE: {ld_rmse:.6f})')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                # Performance error plot
                cl_error = cl_original_arr - cl_reconstructed
                cd_error = cd_original_arr - cd_reconstructed
                valid_mask = ~(np.isnan(cl_error) | np.isnan(cd_error))
                if np.any(valid_mask):
                    axes[1, 2].plot(alpha_original_arr[valid_mask], cl_error[valid_mask], 
                                  'g-', label='Cl error', linewidth=1.5)
                    axes[1, 2].plot(alpha_original_arr[valid_mask], cd_error[valid_mask], 
                                  'm-', label='cd error', linewidth=1.5)
                axes[1, 2].set_xlabel('Alpha (degrees)')
                axes[1, 2].set_ylabel('Error')
                axes[1, 2].set_title('Performance Reconstruction Error')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
                axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
                
                plt.tight_layout()
                output_path = PROJECT_ROOT / "test_pipeline" / f"{airfoil_name}_Re{reynolds:.0f}.png"
                output_path.parent.mkdir(exist_ok=True)
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"  Saved plot to {output_path}")
                
            except Exception as e:
                print(f"  Could not create plot: {e}")
        
        except Exception as e:
            print(f"  Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary statistics
    if all_results:
        print(f"\n{'='*60}")
        print("Summary Statistics:")
        print(f"{'='*60}")
        
        avg_x_rmse = np.mean([r['x_rmse'] for r in all_results])
        avg_x_mae = np.mean([r['x_mae'] for r in all_results])
        avg_y_rmse = np.mean([r['y_rmse'] for r in all_results])
        avg_y_mae = np.mean([r['y_mae'] for r in all_results])
        avg_overall_rmse = np.mean([r['overall_rmse'] for r in all_results])
        avg_overall_mae = np.mean([r['overall_mae'] for r in all_results])
        
        # Performance metrics (filter out NaN values)
        cl_rmse_list = [r['cl_rmse'] for r in all_results if not np.isnan(r['cl_rmse'])]
        cl_mae_list = [r['cl_mae'] for r in all_results if not np.isnan(r['cl_mae'])]
        cd_rmse_list = [r['cd_rmse'] for r in all_results if not np.isnan(r['cd_rmse'])]
        cd_mae_list = [r['cd_mae'] for r in all_results if not np.isnan(r['cd_mae'])]
        ld_rmse_list = [r['ld_rmse'] for r in all_results if not np.isnan(r['ld_rmse'])]
        ld_mae_list = [r['ld_mae'] for r in all_results if not np.isnan(r['ld_mae'])]
        
        avg_cl_rmse = np.mean(cl_rmse_list) if cl_rmse_list else np.nan
        avg_cl_mae = np.mean(cl_mae_list) if cl_mae_list else np.nan
        avg_cd_rmse = np.mean(cd_rmse_list) if cd_rmse_list else np.nan
        avg_cd_mae = np.mean(cd_mae_list) if cd_mae_list else np.nan
        avg_ld_rmse = np.mean(ld_rmse_list) if ld_rmse_list else np.nan
        avg_ld_mae = np.mean(ld_mae_list) if ld_mae_list else np.nan
        
        print(f"Average X RMSE:  {avg_x_rmse:.6f}")
        print(f"Average X MAE:   {avg_x_mae:.6f}")
        print(f"Average Y RMSE:  {avg_y_rmse:.6f}")
        print(f"Average Y MAE:   {avg_y_mae:.6f}")
        print(f"Average Overall RMSE: {avg_overall_rmse:.6f}")
        print(f"Average Overall MAE:  {avg_overall_mae:.6f}")
        print(f"\nPerformance Metrics:")
        print(f"Average Cl RMSE:  {avg_cl_rmse:.6f}")
        print(f"Average Cl MAE:   {avg_cl_mae:.6f}")
        print(f"Average cd RMSE:  {avg_cd_rmse:.6f}")
        print(f"Average cd MAE:   {avg_cd_mae:.6f}")
        print(f"Average L/D RMSE: {avg_ld_rmse:.6f}")
        print(f"Average L/D MAE:  {avg_ld_mae:.6f}")
        
        print(f"\nPer-airfoil results:")
        for result in all_results:
            print(f"  {result['name']:30s} - Re: {result['reynolds']:7.0f} - "
                  f"Y RMSE: {result['y_rmse']:.6f}, Y MAE: {result['y_mae']:.6f} - "
                  f"Cl RMSE: {result['cl_rmse']:.6f}, cd RMSE: {result['cd_rmse']:.6f}")
    
    print(f"\n{'='*60}")
    print("✓ Full pipeline test completed!")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("="*60)
    print("Full Pipeline Test Script")
    print("XFoil Data -> Encode -> Latent Mapper -> Decode -> Compare with .dat")
    print("="*60)
    
    test_full_pipeline()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)

