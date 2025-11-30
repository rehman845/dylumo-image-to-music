"""
End-to-end example: Image → Music Recommendation

This script demonstrates the complete pipeline:
1. Load image
2. Extract features
3. Predict music features
4. Get recommendations
"""

import torch
import numpy as np
from pathlib import Path
import argparse

from src.feature_extractors.optimized_image_extractor import OptimizedImageExtractor
from src.models.fast_mlp import FastMLP
from src.recommender.faiss_recommender import FAISSRecommender


def main():
    parser = argparse.ArgumentParser(description='Dylumo end-to-end example')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--music-index', type=str, default='data/processed/music_index.faiss',
                       help='Path to FAISS index')
    parser.add_argument('--k', type=int, default=10,
                       help='Number of recommendations')
    parser.add_argument('--use-text', action='store_true',
                       help='Use text descriptions (slower but better)')
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Step 1: Extract image features
    print("Step 1: Extracting image features...")
    extractor = OptimizedImageExtractor(device=device, use_text=args.use_text)
    image_features = extractor.extract_features(args.image)
    print(f"✓ Image features extracted: {image_features.shape}\n")
    
    # Step 2: Load model and predict music features
    print("Step 2: Predicting music features...")
    model = FastMLP(input_dim=len(image_features), output_dim=13)
    
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(device)
        predicted_music = model.predict(input_tensor).cpu().numpy().flatten()
    
    print(f"✓ Predicted music features:")
    feature_names = [
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'time_signature', 'duration_ms'
    ]
    for name, value in zip(feature_names, predicted_music):
        print(f"  {name}: {value:.4f}")
    print()
    
    # Step 3: Get recommendations
    print("Step 3: Getting recommendations...")
    recommender = FAISSRecommender()
    recommender.load_index(args.music_index)
    
    recommendations = recommender.recommend(predicted_music, k=args.k)
    
    print(f"✓ Top {args.k} recommendations:")
    print(recommendations.to_string())
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()

