"""
01_prepare_spotify.py - Prepare Spotify Dataset

This script:
1. Loads the Spotify 1M tracks dataset
2. Extracts and normalizes 13 audio features
3. Maps songs to emotion categories based on valence/energy
4. Saves processed data for training

Usage:
    python training/01_prepare_spotify.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
from tqdm import tqdm

# Paths
DATA_DIR = PROJECT_ROOT / "data"
SPOTIFY_RAW = DATA_DIR / "spotify" / "spotify_data.csv"
PROCESSED_DIR = DATA_DIR / "processed"

# The 13 audio features we extract
AUDIO_FEATURES = [
    'danceability',      # 0-1
    'energy',            # 0-1
    'key',               # 0-11 -> normalized
    'loudness',          # -60 to 0 -> normalized
    'mode',              # 0 or 1
    'speechiness',       # 0-1
    'acousticness',      # 0-1
    'instrumentalness',  # 0-1
    'liveness',          # 0-1
    'valence',           # 0-1
    'tempo',             # 0-250 -> normalized
    'duration_ms',       # -> normalized
    'time_signature'     # 3-7 -> normalized
]

# Metadata columns to keep
METADATA_COLS = ['track_id', 'track_name', 'artist_name', 'popularity', 'year', 'genre']

# Emotion mapping based on Russell's Circumplex Model
# High valence + High energy = excitement, amusement
# High valence + Low energy = contentment
# Low valence + High energy = anger, fear
# Low valence + Low energy = sadness
EMOTION_CATEGORIES = ['anger', 'amusement', 'fear', 'sadness', 'excitement', 'awe', 'contentment']


def map_to_emotion(valence: float, energy: float) -> str:
    """
    Map valence and energy to emotion category.
    Based on Russell's Circumplex Model of Affect.
    
    Args:
        valence: Valence score (0-1)
        energy: Energy score (0-1)
        
    Returns:
        Emotion category string
    """
    # Define thresholds
    v_mid, e_mid = 0.5, 0.5
    
    if valence >= v_mid:
        # Positive valence
        if energy >= 0.7:
            return 'excitement'
        elif energy >= e_mid:
            return 'amusement'
        else:
            return 'contentment'
    else:
        # Negative valence
        if energy >= 0.7:
            return 'anger'
        elif energy >= e_mid:
            return 'fear'
        else:
            return 'sadness'


def prepare_spotify_data():
    """Main function to prepare Spotify data."""
    
    print("=" * 60)
    print("DYLUMO - Prepare Spotify Dataset")
    print("=" * 60)
    
    # Check if raw data exists
    if not SPOTIFY_RAW.exists():
        print(f"[ERROR] Spotify data not found at {SPOTIFY_RAW}")
        print("Run: python kaggle/setup_kaggle.py --download")
        return False
    
    # Create processed directory
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n[INFO] Loading Spotify data from {SPOTIFY_RAW}...")
    df = pd.read_csv(SPOTIFY_RAW)
    print(f"[OK] Loaded {len(df):,} tracks")
    
    # Remove duplicates by track_id
    original_count = len(df)
    df = df.drop_duplicates(subset=['track_id'])
    print(f"[INFO] Removed {original_count - len(df):,} duplicates")
    
    # Check for missing values in audio features
    missing = df[AUDIO_FEATURES].isnull().sum()
    if missing.any():
        print(f"[WARNING] Missing values found:")
        print(missing[missing > 0])
        df = df.dropna(subset=AUDIO_FEATURES)
        print(f"[INFO] After dropping NaN: {len(df):,} tracks")
    
    # Extract audio features
    print(f"\n[INFO] Extracting {len(AUDIO_FEATURES)} audio features...")
    features = df[AUDIO_FEATURES].copy()
    
    # Normalize features
    print("[INFO] Normalizing features...")
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    features_normalized = pd.DataFrame(
        features_normalized, 
        columns=AUDIO_FEATURES,
        index=features.index
    )
    
    # Map to emotions
    print("[INFO] Mapping songs to emotion categories...")
    emotions = df.apply(
        lambda row: map_to_emotion(row['valence'], row['energy']),
        axis=1
    )
    
    # Print emotion distribution
    print("\n[INFO] Emotion distribution:")
    emotion_counts = emotions.value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count:,} ({count/len(emotions)*100:.1f}%)")
    
    # Create processed dataframe
    processed = pd.DataFrame({
        'track_id': df['track_id'].values,
        'track_name': df['track_name'].values,
        'artist_name': df['artist_name'].values,
        'popularity': df['popularity'].values,
        'emotion': emotions.values
    })
    
    # Add normalized features
    for col in AUDIO_FEATURES:
        processed[col] = features_normalized[col].values
    
    # Save processed data
    output_path = PROCESSED_DIR / "spotify_processed.parquet"
    processed.to_parquet(output_path, index=False)
    print(f"\n[OK] Saved processed data to {output_path}")
    print(f"    Shape: {processed.shape}")
    
    # Save scaler for inference
    scaler_path = PROCESSED_DIR / "spotify_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"[OK] Saved scaler to {scaler_path}")
    
    # Save just the features as numpy array for FAISS
    features_path = PROCESSED_DIR / "spotify_features.npy"
    np.save(features_path, features_normalized.values.astype(np.float32))
    print(f"[OK] Saved features array to {features_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] Spotify data preparation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    print(f"  - {scaler_path}")
    print(f"  - {features_path}")
    print(f"\nNext step: python training/02_prepare_emid.py")
    
    return True


if __name__ == "__main__":
    prepare_spotify_data()

