"""
Kaggle Training Script for Dylumo

Optimized for Kaggle's GPU environment.
Run this in a Kaggle notebook with GPU enabled.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm
import sys

# Add src to path (Kaggle structure)
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/input')

# Import our modules
from src.models.fast_mlp import FastMLP, WeightedMSELoss


class ImageMusicDataset(Dataset):
    """Dataset for image-music pairs."""
    
    def __init__(self, image_features: np.ndarray, music_features: np.ndarray):
        self.image_features = torch.FloatTensor(image_features)
        self.music_features = torch.FloatTensor(music_features)
    
    def __len__(self):
        return len(self.image_features)
    
    def __getitem__(self, idx):
        return self.image_features[idx], self.music_features[idx]


def train_epoch(model, dataloader, criterion, optimizer, device, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    scaler = torch.cuda.amp.GradScaler() if use_amp and device.type == 'cuda' else None
    
    for image_feat, music_feat in tqdm(dataloader, desc="Training"):
        image_feat = image_feat.to(device)
        music_feat = music_feat.to(device)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                pred = model(image_feat)
                loss = criterion(pred, music_feat)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(image_feat)
            loss = criterion(pred, music_feat)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for image_feat, music_feat in tqdm(dataloader, desc="Validating"):
            image_feat = image_feat.to(device)
            music_feat = music_feat.to(device)
            
            pred = model(image_feat)
            loss = criterion(pred, music_feat)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    # Kaggle paths
    INPUT_DIR = Path('/kaggle/input')
    WORKING_DIR = Path('/kaggle/working')
    
    # Configuration
    config = {
        'image_features_path': INPUT_DIR / 'dylumo-data' / 'image_features.npy',
        'music_features_path': INPUT_DIR / 'dylumo-data' / 'music_features.npy',
        'output_dir': WORKING_DIR / 'models',
        'batch_size': 128,  # Larger batch for GPU
        'epochs': 50,
        'lr': 1e-3,
        'input_dim': 1024,
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Load data
    print("\nLoading data...")
    image_features = np.load(config['image_features_path'])
    music_features = np.load(config['music_features_path'])
    
    print(f"  Image features: {image_features.shape}")
    print(f"  Music features: {music_features.shape}")
    
    # Split train/val (80/20)
    n_train = int(0.8 * len(image_features))
    train_img = image_features[:n_train]
    train_music = music_features[:n_train]
    val_img = image_features[n_train:]
    val_music = music_features[n_train:]
    
    print(f"\nTrain: {len(train_img):,} pairs")
    print(f"Val: {len(val_img):,} pairs")
    
    # Create datasets
    train_dataset = ImageMusicDataset(train_img, train_music)
    val_dataset = ImageMusicDataset(val_img, val_music)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Create model
    print("\nCreating model...")
    model = FastMLP(input_dim=config['input_dim'], output_dim=13)
    model = model.to(device)
    
    # Mixed precision for GPU
    if device.type == 'cuda':
        model = model.half()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    
    # Training loop
    output_dir = config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    for epoch in range(config['epochs']):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': config
            }
            torch.save(checkpoint, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        print()
    
    # Save final model and training history
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': config
    }, output_dir / 'final_model.pth')
    
    print("="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Models saved to: {output_dir}")
    print("="*50)


if __name__ == "__main__":
    main()

