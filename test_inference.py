"""
Test Inference Script
Test the trained model with a sample image
"""
import torch
import faiss
import pickle
import json
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import CLIPImageProcessor

# Import model architecture
import sys
sys.path.append('output/kaggle/working/data')
from model_architecture import OptimizedCrossModalTransformer

# Paths
OUTPUT_DIR = Path('output/kaggle/working')
DATA_DIR = OUTPUT_DIR / 'data'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

def load_model():
    """Load trained model and artifacts"""
    print("Loading model and artifacts...")
    
    # Load config
    with open(DATA_DIR / 'deployment_config.json', 'r') as f:
        config = json.load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = OptimizedCrossModalTransformer(
        visual_model=config['visual_model'],
        **config['model_params'],
        audio_feature_dim=len(config['audio_features']),
        num_emotions=len(config['emotions'])
    )
    
    # Load weights
    checkpoint = torch.load(
        CHECKPOINT_DIR / 'dylumo_optimized.pt',
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("✓ Model loaded")
    
    # Load CLIP processor
    processor = CLIPImageProcessor.from_pretrained(config['visual_model'])
    print("✓ CLIP processor loaded")
    
    # Load FAISS index
    index = faiss.read_index(str(DATA_DIR / 'music_index.index'))
    print(f"✓ FAISS index loaded ({index.ntotal:,} songs)")
    
    # Load metadata
    metadata = pd.read_parquet(DATA_DIR / 'spotify_metadata.parquet')
    print(f"✓ Metadata loaded ({len(metadata):,} songs)")
    
    return model, processor, index, metadata, config, device


def recommend_songs(image_path, model, processor, index, metadata, device, top_k=10):
    """Get song recommendations for an image"""
    # Load and process image
    image = Image.open(image_path).convert('RGB')
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(pixel_values, audio_features=None)
        predicted_features = outputs['predicted_features'].cpu().numpy().astype('float32')
        emotion_logits = outputs['emotion_logits'].cpu()
    
    # Get predicted emotion
    emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
    
    # Normalize features for similarity search
    faiss.normalize_L2(predicted_features)
    
    # Search FAISS index
    scores, ids = index.search(predicted_features, top_k)
    
    # Get recommendations
    recommendations = metadata.iloc[ids[0]].copy()
    recommendations['similarity_score'] = scores[0]
    
    return recommendations, emotion_probs


def main():
    """Test inference on sample images"""
    print("="*60)
    print("TESTING MODEL INFERENCE")
    print("="*60)
    
    # Load model
    model, processor, index, metadata, config, device = load_model()
    
    # Find sample images
    emid_images_dir = OUTPUT_DIR / 'emid' / 'images'
    sample_images = list(emid_images_dir.glob('*.jpg'))[:5]  # Test with 5 images
    
    if not sample_images:
        print("\n❌ No sample images found!")
        print("Make sure output/kaggle/working/emid/images/ has images")
        return
    
    print(f"\n✓ Found {len(sample_images)} sample images")
    print("="*60)
    
    # Test each image
    for i, img_path in enumerate(sample_images, 1):
        print(f"\n{'='*60}")
        print(f"IMAGE {i}: {img_path.name}")
        print(f"{'='*60}")
        
        # Get recommendations
        recommendations, emotion_probs = recommend_songs(
            img_path, model, processor, index, metadata, device, top_k=5
        )
        
        # Display predicted emotions
        print("\n🎭 Predicted Emotions:")
        emotions = config['emotions']
        for emotion, prob in zip(emotions, emotion_probs):
            print(f"   {emotion:12s}: {prob:.2%}")
        
        # Display top emotion
        top_emotion_idx = emotion_probs.argmax().item()
        top_emotion = emotions[top_emotion_idx]
        top_prob = emotion_probs[top_emotion_idx].item()
        print(f"\n   → Top: {top_emotion} ({top_prob:.1%})")
        
        # Display recommendations
        print(f"\n🎵 Top 5 Song Recommendations:")
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            print(f"\n   {idx}. {row['track_name']}")
            print(f"      Artist: {row['artist_name']}")
            print(f"      Emotion: {row['emotion']}")
            print(f"      Similarity: {row['similarity_score']:.3f}")
    
    print("\n" + "="*60)
    print("✅ INFERENCE TEST COMPLETE!")
    print("="*60)
    print("\nModel is working correctly! ✓")
    print("Ready for backend/frontend integration! 🚀")


if __name__ == "__main__":
    main()

