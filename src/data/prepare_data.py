"""
Data preparation scripts for Dylumo

Handles loading and preprocessing of:
- Spotify 1M dataset (subset to 100K)
- Image datasets
- Creating training pairs
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import random


def load_spotify_dataset(
    csv_path: str,
    n_samples: int = 100000,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Load Spotify dataset and sample subset.
    
    Args:
        csv_path: Path to Spotify CSV file
        n_samples: Number of songs to sample (default: 100K)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with sampled songs
    """
    print(f"Loading Spotify dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"  Total songs: {len(df):,}")
    
    # Sample subset
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=random_seed)
        print(f"  Sampled {n_samples:,} songs")
    
    return df


def get_music_feature_columns() -> List[str]:
    """
    Get list of music feature column names from Spotify dataset.
    
    Returns:
        List of 13 feature column names
    """
    return [
        'danceability',
        'energy',
        'key',
        'loudness',
        'mode',
        'speechiness',
        'acousticness',
        'instrumentalness',
        'liveness',
        'valence',
        'tempo',
        'time_signature',
        'duration_ms'
    ]


def normalize_music_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize music features to [0, 1] range.
    
    Args:
        features: Array of shape (N, 13) with raw features
        
    Returns:
        Normalized features
    """
    # Min-max normalization per feature
    feature_min = features.min(axis=0, keepdims=True)
    feature_max = features.max(axis=0, keepdims=True)
    
    # Avoid division by zero
    feature_range = feature_max - feature_min
    feature_range[feature_range == 0] = 1
    
    normalized = (features - feature_min) / feature_range
    return normalized


def create_training_pairs(
    image_features: np.ndarray,
    image_ids: List[str],
    music_features: np.ndarray,
    music_ids: List[str],
    n_pairs: Optional[int] = None,
    strategy: str = "random"
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Create training pairs from image and music features.
    
    Args:
        image_features: Array of image features (N, feature_dim)
        image_ids: List of image identifiers
        music_features: Array of music features (M, 13)
        music_ids: List of music track identifiers
        n_pairs: Number of pairs to create (None = all combinations)
        strategy: "random" or "similar" (for similar mood matching)
        
    Returns:
        Tuple of (image_features, music_features, pair_ids)
    """
    print(f"Creating training pairs...")
    print(f"  Images: {len(image_features)}")
    print(f"  Songs: {len(music_features)}")
    
    if strategy == "random":
        # Random pairing
        if n_pairs is None:
            n_pairs = len(image_features) * 10  # 10 songs per image
        
        pairs = []
        for _ in range(n_pairs):
            img_idx = random.randint(0, len(image_features) - 1)
            music_idx = random.randint(0, len(music_features) - 1)
            pairs.append((img_idx, music_idx))
        
        # Extract features
        img_feat_pairs = image_features[[p[0] for p in pairs]]
        music_feat_pairs = music_features[[p[1] for p in pairs]]
        pair_ids = [(image_ids[p[0]], music_ids[p[1]]) for p in pairs]
        
    elif strategy == "similar":
        # TODO: Implement similarity-based pairing
        # For now, use random
        return create_training_pairs(
            image_features, image_ids, music_features, music_ids,
            n_pairs, strategy="random"
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    print(f"  Created {len(pairs):,} pairs")
    
    return img_feat_pairs, music_feat_pairs, pair_ids


def prepare_training_data(
    spotify_csv: str,
    image_dir: Optional[str] = None,
    n_songs: int = 100000,
    n_pairs: Optional[int] = None,
    output_dir: str = "data/processed"
) -> dict:
    """
    Main function to prepare all training data.
    
    Args:
        spotify_csv: Path to Spotify dataset CSV
        image_dir: Directory with training images (optional)
        n_songs: Number of songs to use
        n_pairs: Number of training pairs to create
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with prepared data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load Spotify data
    music_df = load_spotify_dataset(spotify_csv, n_samples=n_songs)
    
    # Extract music features
    feature_cols = get_music_feature_columns()
    music_features = music_df[feature_cols].values
    music_ids = music_df['track_id'].tolist() if 'track_id' in music_df.columns else list(range(len(music_df)))
    
    # Normalize features
    music_features = normalize_music_features(music_features)
    
    # Save music data
    np.save(output_path / "music_features.npy", music_features)
    music_df.to_csv(output_path / "music_metadata.csv", index=False)
    
    print(f"\n✓ Music data prepared:")
    print(f"  Features shape: {music_features.shape}")
    print(f"  Saved to: {output_path}")
    
    result = {
        'music_features': music_features,
        'music_metadata': music_df,
        'music_ids': music_ids,
        'feature_columns': feature_cols
    }
    
    # If images provided, create pairs
    if image_dir:
        # This would require image feature extraction
        # For now, return music data only
        print("\n⚠ Image processing not yet implemented")
        print("  Use OptimizedImageExtractor to extract image features first")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("Data preparation module")
    print("=" * 50)
    print("\nTo prepare data:")
    print("  from src.data.prepare_data import prepare_training_data")
    print("  data = prepare_training_data('path/to/spotify.csv', n_songs=100000)")

