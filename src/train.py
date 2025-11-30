"""
Training script for Dylumo image-to-music model

Trains the FastMLP model to map image features to music features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm

from src.models.fast_mlp import FastMLP, WeightedMSELoss
from src.utils.caching import CacheManager


class ImageMusicDataset(Dataset):
    """Dataset for image-music pairs."""
    
    def __init__(self, image_features: np.ndarray, music_features: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            image_features: Array of image features (N, feature_dim)
            music_features: Array of music features (N, 13)
        """
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
    parser = argparse.ArgumentParser(description='Train Dylumo model')
    parser.add_argument('--image-features', type=str, required=True,
                       help='Path to image features .npy file')
    parser.add_argument('--music-features', type=str, required=True,
                       help='Path to music features .npy file')
    parser.add_argument('--output-dir', type=str, default='models',
                       help='Directory to save model')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--input-dim', type=int, default=1024,
                       help='Input feature dimension')
    parser.add_argument('--device', type=str, default=None,
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    image_features = np.load(args.image_features)
    music_features = np.load(args.music_features)
    
    print(f"  Image features: {image_features.shape}")
    print(f"  Music features: {music_features.shape}")
    
    # Split train/val
    n_train = int(0.8 * len(image_features))
    train_img = image_features[:n_train]
    train_music = music_features[:n_train]
    val_img = image_features[n_train:]
    val_music = music_features[n_train:]
    
    # Create datasets
    train_dataset = ImageMusicDataset(train_img, train_music)
    val_dataset = ImageMusicDataset(val_img, val_music)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = FastMLP(input_dim=args.input_dim, output_dim=13)
    model = model.to(device)
    
    # Mixed precision
    if device.type == 'cuda':
        model = model.half()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = WeightedMSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        
        print()
    
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()

