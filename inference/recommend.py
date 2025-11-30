"""
recommend.py - Get music recommendations from an image

This script takes an image and returns song recommendations
based on the predicted audio features.

Usage:
    python inference/recommend.py --image path/to/image.jpg
    python inference/recommend.py --image path/to/image.jpg --top_k 10
"""

import sys
from pathlib import Path
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

# Paths
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def load_model(device):
    """Load the trained FastMLP model."""
    from ml.model import FastMLP
    
    model_path = CHECKPOINT_DIR / "best_model.pt"
    if not model_path.exists():
        model_path = CHECKPOINT_DIR / "dylumo_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = FastMLP.load(str(model_path), device=device)
    return model


def load_recommender():
    """Load the FAISS recommender."""
    from ml.recommender import MusicRecommender
    
    index_path = CHECKPOINT_DIR / "music_index"
    
    if not (index_path.with_suffix('.index')).exists():
        raise FileNotFoundError(
            f"FAISS index not found. Run: python inference/build_index.py"
        )
    
    recommender = MusicRecommender()
    recommender.load(str(index_path))
    return recommender


def extract_image_features(image_path, device):
    """Extract CLIP features from an image."""
    from ml.extractor import ImageFeatureExtractor
    
    extractor = ImageFeatureExtractor(device=device)
    features = extractor.extract_single(image_path)
    return features


def recommend(image_path: str, top_k: int = 10):
    """Get music recommendations for an image."""
    
    print("=" * 60)
    print("DYLUMO - Music Recommendations")
    print("=" * 60)
    
    # Check image exists
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_path}")
        return None
    
    print(f"\n[INFO] Input image: {image_path}")
    
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Extract image features
    print("\n[INFO] Extracting image features...")
    image_features = extract_image_features(image_path, device)
    print(f"[OK] Image features: {image_features.shape}")
    
    # Load model
    print("\n[INFO] Loading FastMLP model...")
    model = load_model(device)
    print("[OK] Model loaded")
    
    # Predict music features
    print("\n[INFO] Predicting music features...")
    image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(device)
    with torch.no_grad():
        music_features = model(image_tensor).cpu().numpy()
    print(f"[OK] Predicted features: {music_features.shape}")
    
    # Load recommender and get recommendations
    print("\n[INFO] Finding similar songs...")
    recommender = load_recommender()
    recommendations = recommender.recommend(music_features, k=top_k)
    
    # Display results
    print("\n" + "=" * 60)
    print(f"TOP {top_k} RECOMMENDATIONS")
    print("=" * 60)
    
    for rec in recommendations:
        print(f"\n{rec['rank']}. {rec['track_name']}")
        print(f"   Artist: {rec['artist_name']}")
        print(f"   Similarity: {rec['similarity_score']:.4f}")
        if 'emotion' in rec:
            print(f"   Emotion: {rec['emotion']}")
    
    return recommendations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get music recommendations from an image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")
    
    args = parser.parse_args()
    
    recommend(args.image, args.top_k)

