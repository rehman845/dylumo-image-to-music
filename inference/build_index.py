"""
build_index.py - Build FAISS index for music recommendations

This script builds a FAISS index from Spotify audio features
for fast similarity search during inference.

Usage:
    python inference/build_index.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "processed"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def build_index():
    """Build FAISS index from Spotify features."""
    
    print("=" * 60)
    print("DYLUMO - Build FAISS Index")
    print("=" * 60)
    
    # Check for required files
    features_path = DATA_DIR / "spotify_features.npy"
    metadata_path = DATA_DIR / "spotify_processed.parquet"
    
    # Also check checkpoints (from Kaggle training)
    if not features_path.exists():
        features_path = CHECKPOINT_DIR / "spotify_features.npy"
    if not metadata_path.exists():
        metadata_path = CHECKPOINT_DIR / "spotify_metadata.parquet"
    
    if not features_path.exists():
        print(f"[ERROR] Features not found at {features_path}")
        print("Run training first or download from Kaggle output.")
        return False
    
    # Load data
    print(f"\n[INFO] Loading features from {features_path}")
    features = np.load(features_path)
    print(f"[OK] Loaded {features.shape[0]:,} songs with {features.shape[1]} features")
    
    print(f"\n[INFO] Loading metadata from {metadata_path}")
    metadata = pd.read_parquet(metadata_path)
    print(f"[OK] Loaded metadata for {len(metadata):,} songs")
    
    # Build FAISS index
    from ml.recommender import MusicRecommender
    
    recommender = MusicRecommender()
    recommender.build_index(features, metadata)
    
    # Save index
    index_path = CHECKPOINT_DIR / "music_index"
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    recommender.save(str(index_path))
    
    print("\n" + "=" * 60)
    print("[OK] FAISS index built successfully!")
    print("=" * 60)
    print(f"\nSaved to:")
    print(f"  - {index_path}.index")
    print(f"  - {index_path}.parquet")
    print(f"\nNext: python inference/recommend.py --image your_image.jpg")
    
    return True


if __name__ == "__main__":
    build_index()

