"""
02_prepare_emid.py - Prepare EMID Dataset

This script:
1. Loads EMID metadata CSV
2. Downloads images from HuggingFace (if not present)
3. Organizes images by emotion category
4. Saves processed metadata

Usage:
    python training/02_prepare_emid.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import requests
import zipfile
import io
from tqdm import tqdm

# Paths
DATA_DIR = PROJECT_ROOT / "data"
EMID_DIR = DATA_DIR / "emid"
EMID_CSV = EMID_DIR / "EMID_data.csv"
IMAGES_DIR = EMID_DIR / "images"
PROCESSED_DIR = DATA_DIR / "processed"

# HuggingFace dataset URL
HF_DATASET_URL = "https://huggingface.co/datasets/ecnu-aigc/EMID/resolve/main"


def download_emid_images():
    """Download EMID images from HuggingFace."""
    
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        print("[INFO] Images directory already exists, skipping download")
        return True
    
    print("[INFO] Downloading EMID images from HuggingFace...")
    print("[INFO] This may take a while depending on your connection...")
    
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try to download the images archive
    try:
        # The images might be in a zip file or individual folders
        # Check HuggingFace dataset structure
        image_urls = [
            f"{HF_DATASET_URL}/images.zip",
            f"{HF_DATASET_URL}/data/images.zip",
        ]
        
        for url in image_urls:
            try:
                print(f"[INFO] Trying {url}...")
                response = requests.get(url, stream=True, timeout=30)
                if response.status_code == 200:
                    print("[INFO] Downloading images archive...")
                    total_size = int(response.headers.get('content-length', 0))
                    
                    # Download with progress bar
                    content = io.BytesIO()
                    with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            content.write(chunk)
                            pbar.update(len(chunk))
                    
                    # Extract
                    print("[INFO] Extracting images...")
                    content.seek(0)
                    with zipfile.ZipFile(content, 'r') as zip_ref:
                        zip_ref.extractall(IMAGES_DIR)
                    
                    print("[OK] Images downloaded and extracted!")
                    return True
            except Exception as e:
                print(f"[WARNING] Failed with {url}: {e}")
                continue
        
        print("[WARNING] Could not download images automatically.")
        print("[INFO] Please download images manually from:")
        print("       https://huggingface.co/datasets/ecnu-aigc/EMID")
        print(f"       and extract to: {IMAGES_DIR}")
        return False
        
    except Exception as e:
        print(f"[ERROR] Failed to download images: {e}")
        return False


def prepare_emid_data():
    """Main function to prepare EMID data."""
    
    print("=" * 60)
    print("DYLUMO - Prepare EMID Dataset")
    print("=" * 60)
    
    # Check if CSV exists
    if not EMID_CSV.exists():
        print(f"[ERROR] EMID CSV not found at {EMID_CSV}")
        print("Run: python kaggle/setup_kaggle.py --download")
        return False
    
    # Create directories
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load EMID metadata
    print(f"\n[INFO] Loading EMID metadata from {EMID_CSV}...")
    df = pd.read_csv(EMID_CSV)
    print(f"[OK] Loaded {len(df):,} music-image pairs")
    
    # Extract unique images with their emotions
    print("\n[INFO] Extracting image-emotion pairs...")
    
    image_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        # Each row has 3 images
        for i in [1, 2, 3]:
            filename_col = f'Image{i}_filename'
            tag_col = f'Image{i}_tag'
            text_col = f'Image{i}_text'
            
            if pd.notna(row[filename_col]):
                image_data.append({
                    'image_filename': row[filename_col],
                    'emotion': row[tag_col],
                    'description': row[text_col] if pd.notna(row[text_col]) else '',
                    'music_genre': row['genre'],  # A-M category
                    'is_original': row['is_original_clip']
                })
    
    # Create dataframe and remove duplicates
    images_df = pd.DataFrame(image_data)
    original_count = len(images_df)
    images_df = images_df.drop_duplicates(subset=['image_filename'])
    print(f"[INFO] Found {len(images_df):,} unique images (removed {original_count - len(images_df):,} duplicates)")
    
    # Emotion distribution
    print("\n[INFO] Emotion distribution:")
    emotion_counts = images_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"  {emotion}: {count:,} ({count/len(images_df)*100:.1f}%)")
    
    # Try to download images
    print("\n[INFO] Checking for images...")
    download_emid_images()
    
    # Check which images exist locally
    images_exist = []
    for _, row in images_df.iterrows():
        # Check multiple possible locations
        possible_paths = [
            IMAGES_DIR / row['image_filename'],
            IMAGES_DIR / row['emotion'] / row['image_filename'],
            EMID_DIR / "images" / row['image_filename'],
        ]
        
        exists = any(p.exists() for p in possible_paths)
        images_exist.append(exists)
    
    images_df['exists'] = images_exist
    existing_count = sum(images_exist)
    
    if existing_count > 0:
        print(f"\n[OK] Found {existing_count:,} images locally")
    else:
        print(f"\n[WARNING] No images found locally!")
        print("[INFO] Images need to be downloaded from HuggingFace:")
        print("       https://huggingface.co/datasets/ecnu-aigc/EMID")
        print(f"       Extract images to: {IMAGES_DIR}")
    
    # Save processed metadata
    output_path = PROCESSED_DIR / "emid_images.parquet"
    images_df.to_parquet(output_path, index=False)
    print(f"\n[OK] Saved image metadata to {output_path}")
    print(f"    Shape: {images_df.shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print("[OK] EMID data preparation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {output_path}")
    
    if existing_count == 0:
        print("\n[ACTION REQUIRED] Download images manually:")
        print("1. Go to: https://huggingface.co/datasets/ecnu-aigc/EMID")
        print(f"2. Download images and extract to: {IMAGES_DIR}")
        print("3. Re-run this script to verify")
    else:
        print(f"\nNext step: python training/03_extract_features.py")
    
    return True


if __name__ == "__main__":
    prepare_emid_data()

