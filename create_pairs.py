"""Create training pairs from image and music features"""

from src.data.prepare_data import create_training_pairs
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    # Load features
    image_path = Path('data/processed/image_features.npy')
    music_path = Path('data/processed/music_features.npy')
    
    if not image_path.exists():
        print(f"ERROR: {image_path} not found!")
        print("Run extract_image_features.py first")
        exit(1)
    
    if not music_path.exists():
        print(f"ERROR: {music_path} not found!")
        print("Run prepare_music.py first")
        exit(1)
    
    print("Loading features...")
    image_features = np.load(image_path)
    music_features = np.load(music_path)
    
    print(f"  Image features: {image_features.shape}")
    print(f"  Music features: {music_features.shape}")
    
    # Create pairs
    print("\nCreating training pairs...")
    n_pairs = min(50000, len(image_features) * 10)  # 10 songs per image, max 50K
    
    img_pairs, music_pairs, pair_ids = create_training_pairs(
        image_features=image_features,
        image_ids=[f'img_{i}' for i in range(len(image_features))],
        music_features=music_features,
        music_ids=[f'song_{i}' for i in range(len(music_features))],
        n_pairs=n_pairs,
        strategy='random'
    )
    
    # Save
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / 'train_image_features.npy', img_pairs)
    np.save(output_dir / 'train_music_features.npy', music_pairs)
    
    print(f"\n✓ Training pairs created!")
    print(f"  Created {len(img_pairs)} pairs")
    print(f"  Saved to: data/processed/train_image_features.npy")
    print(f"  Saved to: data/processed/train_music_features.npy")

