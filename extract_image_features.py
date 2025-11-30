"""Extract image features from training images"""

from src.feature_extractors.optimized_image_extractor import OptimizedImageExtractor
import numpy as np
from pathlib import Path

if __name__ == "__main__":
    image_dir = Path('data/images/train')
    
    if not image_dir.exists():
        print(f"ERROR: {image_dir} not found!")
        print("Please create the directory and add training images")
        exit(1)
    
    # Find all images
    image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png')) + list(image_dir.glob('*.jpeg'))
    
    if len(image_paths) == 0:
        print(f"ERROR: No images found in {image_dir}")
        exit(1)
    
    print(f"Found {len(image_paths)} images")
    print("Initializing feature extractor...")
    
    extractor = OptimizedImageExtractor(use_text=True)
    
    print("Extracting features...")
    image_features = []
    
    for i, img_path in enumerate(image_paths):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(image_paths)} images...")
        
        feat = extractor.extract_features(img_path, image_id=img_path.stem, use_cache=True)
        image_features.append(feat)
    
    image_features = np.array(image_features)
    
    # Save
    output_path = Path('data/processed/image_features.npy')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, image_features)
    
    print(f"\n✓ Image features extracted!")
    print(f"  Saved {len(image_features)} features to: {output_path}")
    print(f"  Feature shape: {image_features.shape}")

