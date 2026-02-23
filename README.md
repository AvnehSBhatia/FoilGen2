# FoilGen2

A neural network system for generating airfoil shapes from aerodynamic performance data. This project uses deep learning to learn the relationship between airfoil performance characteristics (lift, drag, etc.) and geometric shapes, enabling the generation of new airfoil designs based on desired performance metrics.

## Overview

FoilGen2 consists of three main neural network components:

1. **XFoil Autoencoder**: Compresses airfoil performance data (Reynolds number, angle of attack, lift coefficient, drag coefficient, lift-to-drag ratio) into a 16-dimensional latent representation.

2. **Airfoil Autoencoder**: Compresses airfoil coordinate data (x and y coordinates) into a 16-dimensional latent representation.

3. **Latent Mapper**: Maps between the performance latent space and the shape latent space, allowing generation of airfoil shapes from performance data.

The complete pipeline works as follows:
- Performance data → XFoil Encoder → Latent Vector → Latent Mapper → Airfoil Latent Vector → Airfoil Decoder → Airfoil Coordinates

## Project Structure

```
FoilGen2/
├── data/
│   ├── airfoil_data/          # Performance data CSV files
│   ├── bigfoil/                # Airfoil coordinate .dat files
│   └── latent_vectors.csv     # Combined latent vectors for training mapper
├── models/                     # Trained model files
│   ├── xfoil_encoder.pth
│   ├── xfoil_decoder.pth
│   ├── xfoil_scaler.pkl
│   ├── airfoil_encoder.pth
│   ├── airfoil_decoder.pth
│   ├── airfoil_scaler.pkl
│   ├── latent_mapper.pth
│   └── latent_mapper_*_scaler.pkl
├── train/                      # Training scripts
│   ├── train_xfoil_autoencoder.py
│   ├── train_airfoil_autoencoder.py
│   ├── train_latent_mapper.py
│   ├── generate_airfoil_data.py
│   ├── create_latent_vectors_csv.py
│   └── test/                   # Testing scripts
└── test_pipeline/             # Pipeline test outputs
```

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- scikit-learn
- matplotlib
- neuralfoil (for generating performance data)

Install dependencies:
```bash
pip install torch numpy scikit-learn matplotlib neuralfoil
```

## Data Preparation

### 1. Airfoil Coordinate Data

Place airfoil coordinate files (.dat format) in `data/bigfoil/`. Each file should contain:
- First line: airfoil name/description
- Subsequent lines: x y coordinate pairs

The system expects airfoils with 200 points (resampled automatically if needed).

### 2. Performance Data Generation

Generate performance data for airfoils:
```bash
python train/generate_airfoil_data.py
```

This script:
- Reads airfoil coordinates from `data/bigfoil/`
- Generates performance data using neuralfoil for multiple Reynolds numbers and angles of attack
- Saves CSV files in `data/airfoil_data/`
- Optionally encodes airfoils to latent vectors if models are available

## Training

Train models in the following order:

### Step 1: Train XFoil Autoencoder

Trains an autoencoder on performance data (Reynolds, alpha, Cl, cd, L/D):
```bash
python train/train_xfoil_autoencoder.py
```

Input: 101 dimensions (1 Reynolds + 25 alpha + 25 Cl + 25 cd + 25 L/D)
Output: 16-dimensional latent vector
Architecture: 101 → 128 → 64 → 32 → 16 → 16 → 32 → 64 → 128 → 101

Saves:
- `models/xfoil_encoder.pth`
- `models/xfoil_decoder.pth`
- `models/xfoil_scaler.pkl`

### Step 2: Train Airfoil Autoencoder

Trains an autoencoder on airfoil coordinate data:
```bash
python train/train_airfoil_autoencoder.py
```

Input: 400 dimensions (200 x-coordinates + 200 y-coordinates)
Output: 16-dimensional latent vector
Architecture: 400 → 256 → 128 → 32 → 16 → 32 → 128 → 256 → 400

Saves:
- `models/airfoil_encoder.pth`
- `models/airfoil_decoder.pth`
- `models/airfoil_scaler.pkl`

### Step 3: Create Latent Vectors CSV

Creates a CSV file with paired latent vectors for training the mapper:
```bash
python train/create_latent_vectors_csv.py
```

This script:
- Loads both encoders
- Encodes performance data and airfoil coordinates to latent vectors
- Creates `data/latent_vectors.csv` with paired xfoil and airfoil latent vectors

### Step 4: Train Latent Mapper

Trains a neural network to map from performance latent space to shape latent space:
```bash
python train/train_latent_mapper.py
```

Input: 16-dimensional xfoil latent vector
Output: 16-dimensional airfoil latent vector
Architecture: 16 → 128 → 256 → 128 → 16

Saves:
- `models/latent_mapper.pth`
- `models/latent_mapper_xfoil_scaler.pkl`
- `models/latent_mapper_airfoil_scaler.pkl`

## Usage

### Testing Individual Components

Test the XFoil autoencoder:
```bash
python train/test/test_xfoil_autoencoder.py
```

Test the airfoil autoencoder:
```bash
python train/test/test_airfoil_autoencoder.py
```

### Full Pipeline Testing

Test the complete pipeline (performance data → airfoil shape):
```bash
python train/test/test_full_pipeline.py
```

Test with a specific airfoil:
```bash
python train/test/test_specific.py
```

The test scripts will:
1. Load performance data for an airfoil
2. Encode it to latent space using the XFoil encoder
3. Map it to airfoil latent space using the latent mapper
4. Decode it to coordinates using the airfoil decoder
5. Compare the generated shape with the original airfoil
6. Save visualization plots

## Architecture Details

### XFoil Autoencoder
- Encoder: 101 → 128 → 64 → 32 → 16 (latent)
- Decoder: 16 → 32 → 64 → 128 → 101
- Activation: Tanh
- Loss: MSE
- Optimizer: Adam

### Airfoil Autoencoder
- Encoder: 400 → 256 → 128 → 32 → 16 (latent)
- Decoder: 16 → 32 → 128 → 256 → 400
- Activation: Tanh
- Loss: MSE
- Optimizer: Adam

### Latent Mapper
- Architecture: 16 → 128 → 256 → 128 → 16
- Activation: Tanh
- Loss: MSE
- Optimizer: Adam with learning rate scheduling

## Configuration

Key parameters can be modified in each training script:

- `LATENT_DIM`: Dimensionality of latent space (default: 16)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 200)
- `LEARNING_RATE`: Learning rate for optimizer
- `TARGET_POINTS`: Number of points per airfoil (default: 200)

## Data Format

### Performance Data CSV
Columns:
- `airfoil_name`: Name of the airfoil
- `reynolds`: Reynolds number
- `alpha`: Angle of attack (degrees)
- `cl`: Lift coefficient
- `cd`: Drag coefficient
- `ld`: Lift-to-drag ratio
- `xfoil_latent_vector`: JSON array of latent vector (if encoded)

### Latent Vectors CSV
Columns:
- `airfoil_name`: Name of the airfoil
- `xfoil_latent_vector`: JSON array of performance latent vector
- `airfoil_latent_vector`: JSON array of shape latent vector

### Airfoil Coordinate File (.dat)
Format:
```
AIRFOIL_NAME
x1 y1
x2 y2
...
```

## Notes

- The system automatically resamples airfoils to 200 points if they have a different number
- All models use GPU if available, otherwise CPU
- Data is normalized using StandardScaler before training
- Models are saved with architecture information for easy loading
- The latent space dimensionality (16) was chosen to balance compression and reconstruction quality

## Troubleshooting

**Issue**: Models not found when running tests
- Solution: Ensure all training steps have been completed and models are in the `models/` directory

**Issue**: CUDA out of memory
- Solution: Reduce `BATCH_SIZE` in the training scripts

**Issue**: Airfoil files not loading
- Solution: Check that .dat files are in the correct format and located in `data/bigfoil/`

**Issue**: Performance data generation fails
- Solution: Ensure neuralfoil is installed and airfoil coordinates are valid

