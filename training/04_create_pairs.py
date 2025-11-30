"""
04_create_pairs.py - Create Image-Music Training Pairs

This script:
1. Loads processed image features and Spotify data
2. Matches images to songs by emotion category
3. Creates training pairs (image_features -> music_features)
4. Saves pairs for training

Usage:
    python training/04_create_pairs.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from tqdm import tqdm
import random

# Paths
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# Pairing settings
SONGS_PER_IMAGE = 5  # Number of songs to pair with each image
RANDOM_SEED = 42


def create_training_pairs():
    """Main function to create training pairs."""
    
    print("=" * 60)
    print("DYLUMO - Create Training Pairs")
    print("=" * 60)
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # Check for required files
    image_features_path = PROCESSED_DIR / "image_features.npy"
    image_metadata_path = PROCESSED_DIR / "image_metadata.parquet"
    spotify_path = PROCESSED_DIR / "spotify_processed.parquet"
    
    for path in [image_features_path, image_metadata_path, spotify_path]:
        if not path.exists():
            print(f"[ERROR] Required file not found: {path}")
            print("Run previous preparation scripts first.")
            return False
    
    # Load data
    print("\n[INFO] Loading data...")
    
    image_features = np.load(image_features_path)
    print(f"[OK] Image features: {image_features.shape}")
    
    image_meta = pd.read_parquet(image_metadata_path)
    print(f"[OK] Image metadata: {len(image_meta):,} images")
    
    spotify_df = pd.read_parquet(spotify_path)
    print(f"[OK] Spotify data: {len(spotify_df):,} songs")
    
    # Get audio feature columns
    from training import AUDIO_FEATURES
    audio_feature_cols = AUDIO_FEATURES
    
    # Verify columns exist
    missing_cols = [c for c in audio_feature_cols if c not in spotify_df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns in Spotify data: {missing_cols}")
        return False
    
    # Group Spotify songs by emotion
    print("\n[INFO] Grouping songs by emotion...")
    emotion_songs = {}
    for emotion in image_meta['emotion'].unique():
        songs = spotify_df[spotify_df['emotion'] == emotion]
        emotion_songs[emotion] = songs
        print(f"  {emotion}: {len(songs):,} songs")
    
    # Create pairs
    print(f"\n[INFO] Creating pairs ({SONGS_PER_IMAGE} songs per image)...")
    
    pair_image_features = []
    pair_music_features = []
    pair_metadata = []
    
    for idx, (img_idx, row) in enumerate(tqdm(image_meta.iterrows(), total=len(image_meta), desc="Creating pairs")):
        emotion = row['emotion']
        
        # Get songs with matching emotion
        available_songs = emotion_songs.get(emotion)
        
        if available_songs is None or len(available_songs) == 0:
            continue
        
        # Sample songs for this image
        n_samples = min(SONGS_PER_IMAGE, len(available_songs))
        sampled_songs = available_songs.sample(n=n_samples, random_state=RANDOM_SEED + idx)
        
        # Create pairs
        for _, song in sampled_songs.iterrows():
            # Image features
            pair_image_features.append(image_features[idx])
            
            # Music features (13-dim normalized audio features)
            music_feat = song[audio_feature_cols].values.astype(np.float32)
            pair_music_features.append(music_feat)
            
            # Metadata for debugging
            pair_metadata.append({
                'image_file': row['image_filename'],
                'image_emotion': emotion,
                'track_id': song['track_id'],
                'track_name': song['track_name'],
                'artist_name': song['artist_name']
            })
    
    # Convert to arrays
    X = np.array(pair_image_features, dtype=np.float32)
    y = np.array(pair_music_features, dtype=np.float32)
    
    print(f"\n[OK] Created {len(X):,} training pairs")
    print(f"    Image features shape: {X.shape}")
    print(f"    Music features shape: {y.shape}")
    
    # Split into train/val/test (80/10/10)
    print("\n[INFO] Splitting data...")
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    
    train_end = int(0.8 * n_samples)
    val_end = int(0.9 * n_samples)
    
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    
    splits = {
        'train': (X[train_idx], y[train_idx]),
        'val': (X[val_idx], y[val_idx]),
        'test': (X[test_idx], y[test_idx])
    }
    
    for split_name, (split_X, split_y) in splits.items():
        print(f"  {split_name}: {len(split_X):,} pairs")
    
    # Save pairs
    print("\n[INFO] Saving pairs...")
    
    # Save as numpy arrays
    np.save(PROCESSED_DIR / "train_X.npy", splits['train'][0])
    np.save(PROCESSED_DIR / "train_y.npy", splits['train'][1])
    np.save(PROCESSED_DIR / "val_X.npy", splits['val'][0])
    np.save(PROCESSED_DIR / "val_y.npy", splits['val'][1])
    np.save(PROCESSED_DIR / "test_X.npy", splits['test'][0])
    np.save(PROCESSED_DIR / "test_y.npy", splits['test'][1])
    
    # Save metadata
    meta_df = pd.DataFrame(pair_metadata)
    meta_df.to_parquet(PROCESSED_DIR / "pairs_metadata.parquet", index=False)
    
    print("[OK] Saved training data:")
    for split in ['train', 'val', 'test']:
        print(f"  - {split}_X.npy, {split}_y.npy")
    print(f"  - pairs_metadata.parquet")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("[OK] Training pairs created!")
    print("=" * 60)
    print(f"\nDataset statistics:")
    print(f"  Total pairs: {len(X):,}")
    print(f"  Train: {len(splits['train'][0]):,}")
    print(f"  Val:   {len(splits['val'][0]):,}")
    print(f"  Test:  {len(splits['test'][0]):,}")
    print(f"\nFeature dimensions:")
    print(f"  Input (image):  {X.shape[1]}")
    print(f"  Output (music): {y.shape[1]}")
    print(f"\nNext step: python training/05_train.py")
    
    return True




if __name__ == "__main__":
    create_training_pairs()

