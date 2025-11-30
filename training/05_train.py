"""
05_train.py - Train FastMLP Model (Local)

This script trains locally. For faster training with GPU, use Kaggle notebook instead:
    kaggle/train_notebook.ipynb

Usage:
    python training/05_train.py
    python training/05_train.py --epochs 100 --batch_size 128
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# Default training settings
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 0.001


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_data():
    """Load training data."""
    print("[INFO] Loading training data...")
    
    train_X = np.load(PROCESSED_DIR / "train_X.npy")
    train_y = np.load(PROCESSED_DIR / "train_y.npy")
    val_X = np.load(PROCESSED_DIR / "val_X.npy")
    val_y = np.load(PROCESSED_DIR / "val_y.npy")
    
    print(f"[OK] Train: {train_X.shape[0]:,} samples")
    print(f"[OK] Val:   {val_X.shape[0]:,} samples")
    
    return train_X, train_y, val_X, val_y


def train_model(epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, learning_rate=DEFAULT_LR):
    """Main training function."""
    
    print("=" * 60)
    print("DYLUMO - Train FastMLP Model (Local)")
    print("=" * 60)
    print("\nNote: For faster training, use Kaggle GPU:")
    print("      Upload kaggle/train_notebook.ipynb to Kaggle\n")
    
    # Create checkpoint directory
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"[INFO] Using device: {device}")
    
    # Load data
    try:
        train_X, train_y, val_X, val_y = load_data()
    except FileNotFoundError:
        print("[ERROR] Training data not found!")
        print("Run the training pipeline first:")
        print("  python training/01_prepare_spotify.py")
        print("  python training/02_prepare_emid.py")
        print("  python training/03_extract_features.py")
        print("  python training/04_create_pairs.py")
        return False
    
    # Get dimensions
    input_dim = train_X.shape[1]
    output_dim = train_y.shape[1]
    
    print(f"\n[INFO] Input dimension:  {input_dim}")
    print(f"[INFO] Output dimension: {output_dim}")
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(torch.FloatTensor(train_X), torch.FloatTensor(train_y))
    val_dataset = TensorDataset(torch.FloatTensor(val_X), torch.FloatTensor(val_y))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    from ml.model import FastMLP
    model = FastMLP(input_dim=input_dim, output_dim=output_dim, hidden_dims=[512, 256, 128])
    model = model.to(device)
    
    print(f"\n[INFO] Model: FastMLP")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    
    # Training loop
    print(f"\n[INFO] Training for {epochs} epochs...")
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
        
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 10 == 0 or val_loss < best_val_loss:
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save(str(CHECKPOINT_DIR / "best_model.pt"))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= 15:
            print(f"\n[INFO] Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    model.save(str(CHECKPOINT_DIR / "final_model.pt"))
    
    print("\n" + "=" * 60)
    print("[OK] Training complete!")
    print("=" * 60)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"\nSaved models:")
    print(f"  - {CHECKPOINT_DIR / 'best_model.pt'}")
    print(f"  - {CHECKPOINT_DIR / 'final_model.pt'}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FastMLP model")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr)

