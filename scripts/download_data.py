"""
DYLUMO - Download All Datasets Script

This script downloads both datasets to your local data/ folder:
1. Spotify 1M Tracks (from Kaggle)
2. EMID Images (from HuggingFace)

Usage:
    python scripts/download_data.py

Prerequisites:
    - Kaggle API configured (~/.kaggle/kaggle.json)
    - pip install kaggle huggingface_hub pyarrow tqdm
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPOTIFY_DIR = RAW_DIR / "spotify"
EMID_DIR = RAW_DIR / "emid"
IMAGES_DIR = EMID_DIR / "images"


def create_directories():
    """Create all required directories."""
    print("\n[1/5] Creating directories...")
    
    dirs = [DATA_DIR, RAW_DIR, SPOTIFY_DIR, EMID_DIR, IMAGES_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {d}")
    
    print("  Done!")


def download_spotify():
    """Download Spotify dataset from Kaggle."""
    print("\n[2/5] Downloading Spotify dataset from Kaggle...")
    
    spotify_csv = SPOTIFY_DIR / "spotify_data.csv"
    
    if spotify_csv.exists():
        print(f"  Already exists: {spotify_csv}")
        return True
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print("  Downloading spotify-1million-tracks...")
        api.dataset_download_files(
            'amitanshjoshi/spotify-1million-tracks',
            path=str(SPOTIFY_DIR),
            unzip=True
        )
        
        print(f"  Saved to: {SPOTIFY_DIR}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        print("\n  To fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Click 'Create New API Token'")
        print("  3. Save kaggle.json to ~/.kaggle/")
        return False


def download_emid():
    """Download EMID dataset from HuggingFace."""
    print("\n[3/5] Downloading EMID dataset from HuggingFace...")
    
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        from tqdm import tqdm
        
        # Create raw parquet directory
        parquet_dir = EMID_DIR / "parquet"
        parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Download metadata CSV
        print("  Downloading EMID_data.csv...")
        hf_hub_download(
            repo_id='ecnu-aigc/EMID',
            repo_type='dataset',
            filename='EMID_data.csv',
            local_dir=str(EMID_DIR)
        )
        
        # List and download parquet files
        print("  Listing parquet files...")
        files = list_repo_files('ecnu-aigc/EMID', repo_type='dataset')
        parquet_files = [f for f in files if f.endswith('.parquet')]
        
        print(f"  Downloading {len(parquet_files)} parquet files...")
        for pf in tqdm(parquet_files, desc="  Downloading"):
            hf_hub_download(
                repo_id='ecnu-aigc/EMID',
                repo_type='dataset',
                filename=pf,
                local_dir=str(parquet_dir)
            )
        
        print(f"  Saved to: {EMID_DIR}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def extract_images():
    """Extract images from EMID parquet files."""
    print("\n[4/5] Extracting images from parquet files...")
    
    try:
        import pyarrow.parquet as pq
        from tqdm import tqdm
        
        # Find parquet files
        parquet_path = EMID_DIR / "parquet" / "data"
        
        if not parquet_path.exists():
            print(f"  ERROR: Parquet directory not found: {parquet_path}")
            return False
        
        parquet_files = list(parquet_path.glob("*.parquet"))
        
        if not parquet_files:
            print("  ERROR: No parquet files found")
            return False
        
        print(f"  Found {len(parquet_files)} parquet files")
        
        saved_count = 0
        for pq_file in tqdm(parquet_files, desc="  Extracting"):
            try:
                table = pq.read_table(pq_file)
                df = table.to_pandas()
                
                for _, row in df.iterrows():
                    for col in ['Image1_filename', 'Image2_filename', 'Image3_filename']:
                        try:
                            data = row.get(col)
                            if data and isinstance(data, dict) and 'bytes' in data and data['bytes']:
                                filename = os.path.basename(data.get('path', f'{col}.jpg'))
                                filepath = IMAGES_DIR / filename
                                
                                if not filepath.exists():
                                    filepath.write_bytes(data['bytes'])
                                    saved_count += 1
                        except Exception:
                            continue
            except Exception as e:
                print(f"  Warning: Error processing {pq_file}: {e}")
                continue
        
        print(f"  Extracted {saved_count} images to: {IMAGES_DIR}")
        return True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_data():
    """Verify downloaded data."""
    print("\n[5/5] Verifying downloaded data...")
    
    # Check Spotify
    spotify_csv = SPOTIFY_DIR / "spotify_data.csv"
    if spotify_csv.exists():
        import pandas as pd
        df = pd.read_csv(spotify_csv)
        print(f"  Spotify: {len(df):,} tracks")
    else:
        print("  Spotify: NOT FOUND")
    
    # Check EMID metadata
    emid_csv = EMID_DIR / "EMID_data.csv"
    if emid_csv.exists():
        import pandas as pd
        df = pd.read_csv(emid_csv)
        print(f"  EMID metadata: {len(df):,} entries")
    else:
        print("  EMID metadata: NOT FOUND")
    
    # Check images
    if IMAGES_DIR.exists():
        images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
        print(f"  EMID images: {len(images):,} images")
    else:
        print("  EMID images: NOT FOUND")


def main():
    print("=" * 60)
    print("DYLUMO - Download All Datasets")
    print("=" * 60)
    
    # Step 1: Create directories
    create_directories()
    
    # Step 2: Download Spotify
    spotify_ok = download_spotify()
    
    # Step 3: Download EMID
    emid_ok = download_emid()
    
    # Step 4: Extract images
    if emid_ok:
        extract_images()
    
    # Step 5: Verify
    verify_data()
    
    print("\n" + "=" * 60)
    if spotify_ok and emid_ok:
        print("SUCCESS! All datasets downloaded.")
    else:
        print("PARTIAL SUCCESS. Some datasets may be missing.")
    print("=" * 60)
    
    print("\nData structure:")
    print(f"  {DATA_DIR}/")
    print(f"    raw/")
    print(f"      spotify/")
    print(f"        spotify_data.csv")
    print(f"      emid/")
    print(f"        EMID_data.csv")
    print(f"        images/")
    print(f"          *.jpg")


if __name__ == "__main__":
    main()

