"""
Basic Example: Load an image and extract features using CLIP

This is the simplest possible starting point - just to verify everything works.
"""

import torch
import clip
from PIL import Image
import numpy as np

def load_image_features(image_path):
    """
    Load an image and extract features using CLIP.
    
    Args:
        image_path: Path to image file
        
    Returns:
        numpy array: Image embedding vector
    """
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy().flatten()

if __name__ == "__main__":
    print("Dylumo - Basic Example")
    print("=" * 50)
    
    # For now, just test if CLIP loads
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"✓ CLIP loaded successfully on {device}")
    print(f"  Model: ViT-B/32")
    print(f"  Image embedding dimension: 512")
    
    print("\nNext steps:")
    print("1. Add a test image to data/test_images/")
    print("2. Run: python src/example_basic.py <image_path>")
    print("3. You'll see the image embedding vector")

