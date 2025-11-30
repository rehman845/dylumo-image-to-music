"""Prepare music data from Spotify dataset - extracts 100K subset"""

from src.data.prepare_data import prepare_training_data
from pathlib import Path

if __name__ == "__main__":
    spotify_csv = 'data/raw/spotify_tracks.csv'
    
    if not Path(spotify_csv).exists():
        print(f"ERROR: {spotify_csv} not found!")
        print("Please download the Spotify dataset and place it in data/raw/")
        exit(1)
    
    print("Preparing music data (100K subset)...")
    data = prepare_training_data(
        spotify_csv=spotify_csv,
        n_songs=100000,
        output_dir='data/processed'
    )
    
    print("\n✓ Music data prepared!")
    print(f"  Features saved to: data/processed/music_features.npy")
    print(f"  Metadata saved to: data/processed/music_metadata.csv")

