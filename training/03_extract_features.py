"""
03_extract_features.py - Extract Image Features using CLIP

This script:
1. Loads EMID image metadata
2. Extracts CLIP ViT-B/32 features (512-dim) from each image
3. Saves features as numpy array for training

Usage:
    python training/03_extract_features.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EMID_DIR = DATA_DIR / "emid"
IMAGES_DIR = EMID_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

# Feature extraction settings
BATCH_SIZE = 32
FEATURE_DIM = 512  # CLIP ViT-B/32 output dimension


def find_image_path(filename: str, emotion: str = None) -> Path:
    """
    Find the actual path to an image file.
    
    Args:
        filename: Image filename
        emotion: Emotion category (optional, for subfolder search)
        
    Returns:
        Path to image or None if not found
    """
    # Try various possible locations
    possible_paths = [
        IMAGES_DIR / filename,
        IMAGES_DIR / emotion / filename if emotion else None,
        EMID_DIR / filename,
        DATA_DIR / "emid" / "images" / filename,
    ]
    
    for path in possible_paths:
        if path and path.exists():
            return path
    
    return None


def extract_features():
    """Main function to extract image features."""
    
    print("=" * 60)
    print("DYLUMO - Extract Image Features")
    print("=" * 60)
    
    # Check for processed EMID data
    emid_path = PROCESSED_DIR / "emid_images.parquet"
    if not emid_path.exists():
        print(f"[ERROR] EMID metadata not found at {emid_path}")
        print("Run: python training/02_prepare_emid.py")
        return False
    
    # Load EMID metadata
    print(f"\n[INFO] Loading EMID metadata...")
    df = pd.read_parquet(emid_path)
    print(f"[OK] Loaded {len(df):,} image records")
    
    # Filter to only existing images
    if 'exists' in df.columns:
        df_exist = df[df['exists'] == True].copy()
    else:
        # Check existence now
        exists = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Checking images"):
            path = find_image_path(row['image_filename'], row.get('emotion'))
            exists.append(path is not None)
        df['exists'] = exists
        df_exist = df[df['exists'] == True].copy()
    
    if len(df_exist) == 0:
        print("[ERROR] No images found!")
        print("Please download EMID images from HuggingFace first.")
        return False
    
    print(f"[INFO] Found {len(df_exist):,} images to process")
    
    # Initialize feature extractor
    print("\n[INFO] Loading CLIP model...")
    from ml.extractor import ImageFeatureExtractor
    extractor = ImageFeatureExtractor(use_blip=False)
    print(f"[OK] Feature dimension: {extractor.get_feature_dim()}")
    
    # Extract features
    print(f"\n[INFO] Extracting features (batch_size={BATCH_SIZE})...")
    
    all_features = []
    valid_indices = []
    
    # Process in batches
    for start_idx in tqdm(range(0, len(df_exist), BATCH_SIZE), desc="Extracting"):
        end_idx = min(start_idx + BATCH_SIZE, len(df_exist))
        batch_df = df_exist.iloc[start_idx:end_idx]
        
        # Load images
        batch_images = []
        batch_indices = []
        
        for idx, row in batch_df.iterrows():
            img_path = find_image_path(row['image_filename'], row.get('emotion'))
            if img_path:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_images.append(img)
                    batch_indices.append(idx)
                except Exception as e:
                    print(f"[WARNING] Could not load {img_path}: {e}")
        
        if batch_images:
            # Extract features for batch
            try:
                features = extractor.extract_batch(batch_images, batch_size=len(batch_images), show_progress=False)
                all_features.append(features)
                valid_indices.extend(batch_indices)
            except Exception as e:
                print(f"[WARNING] Batch extraction failed: {e}")
    
    if not all_features:
        print("[ERROR] No features extracted!")
        return False
    
    # Combine all features
    features_array = np.vstack(all_features).astype(np.float32)
    print(f"\n[OK] Extracted features shape: {features_array.shape}")
    
    # Create filtered dataframe with valid indices
    df_valid = df_exist.loc[valid_indices].reset_index(drop=True)
    
    # Save features
    features_path = PROCESSED_DIR / "image_features.npy"
    np.save(features_path, features_array)
    print(f"[OK] Saved features to {features_path}")
    
    # Save updated metadata with valid images only
    metadata_path = PROCESSED_DIR / "image_metadata.parquet"
    df_valid.to_parquet(metadata_path, index=False)
    print(f"[OK] Saved metadata to {metadata_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] Feature extraction complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {features_path} ({features_array.shape})")
    print(f"  - {metadata_path}")
    print(f"\nFeature statistics:")
    print(f"  - Mean: {features_array.mean():.4f}")
    print(f"  - Std:  {features_array.std():.4f}")
    print(f"  - Min:  {features_array.min():.4f}")
    print(f"  - Max:  {features_array.max():.4f}")
    print(f"\nNext step: python training/04_create_pairs.py")
    
    return True


if __name__ == "__main__":
    extract_features()

