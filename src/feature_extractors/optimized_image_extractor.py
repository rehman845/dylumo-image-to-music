"""
Optimized Image Feature Extractor

Uses CLIP ViT-B/32 for fast, high-quality image embeddings.
Supports cached text descriptions via BLIP-2 for enhanced semantics.
"""

import torch
import clip
import numpy as np
from PIL import Image
from pathlib import Path
import pickle
import json
from typing import Union, Optional, Tuple
from tqdm import tqdm
import os


class OptimizedImageExtractor:
    """
    Fast image feature extractor using CLIP ViT-B/32.
    Supports text description caching for improved performance.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        use_text: bool = True,
        cache_dir: str = "data/cache"
    ):
        """
        Initialize the image extractor.
        
        Args:
            device: 'cuda', 'cpu', or None (auto-detect)
            use_text: Whether to include text embeddings (requires BLIP-2)
            cache_dir: Directory to cache text descriptions
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_text = use_text
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load CLIP model (ViT-B/32 for speed)
        print(f"Loading CLIP ViT-B/32 on {self.device}...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Enable mixed precision for faster inference
        if self.device == "cuda":
            self.model = self.model.half()  # FP16
        
        print(f"✓ CLIP loaded successfully")
        print(f"  Model: ViT-B/32")
        print(f"  Image embedding dimension: 512")
        
        # Load BLIP-2 if text descriptions are needed
        self.blip_model = None
        self.blip_processor = None
        if use_text:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                print("Loading BLIP-2 for text descriptions...")
                self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.blip_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base"
                ).to(self.device)
                self.blip_model.eval()
                print("✓ BLIP-2 loaded successfully")
            except ImportError:
                print("⚠ Warning: transformers not installed. Text descriptions disabled.")
                self.use_text = False
    
    def extract_image_embedding(self, image: Union[Image.Image, str, Path]) -> np.ndarray:
        """
        Extract image embedding using CLIP.
        
        Args:
            image: PIL Image, image path, or Path object
            
        Returns:
            numpy array: 512-dim image embedding (normalized)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Preprocess and encode
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                image_tensor = image_tensor.half()  # FP16
            
            image_features = self.model.encode_image(image_tensor)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().flatten()
    
    def generate_text_description(self, image: Union[Image.Image, str, Path]) -> str:
        """
        Generate text description using BLIP-2.
        
        Args:
            image: PIL Image, image path, or Path object
            
        Returns:
            str: Text description
        """
        if not self.use_text or self.blip_model is None:
            return ""
        
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        
        # Generate caption
        inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            out = self.blip_model.generate(**inputs, max_length=50)
        
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def get_cached_text_embedding(self, image_id: str) -> Optional[np.ndarray]:
        """Load cached text embedding if available."""
        cache_file = self.cache_dir / f"{image_id}_text_emb.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_text_embedding(self, image_id: str, text_emb: np.ndarray):
        """Cache text embedding for future use."""
        cache_file = self.cache_dir / f"{image_id}_text_emb.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(text_emb, f)
    
    def extract_text_embedding(self, text: str) -> np.ndarray:
        """
        Extract text embedding using CLIP text encoder.
        
        Args:
            text: Text description
            
        Returns:
            numpy array: 512-dim text embedding (normalized)
        """
        text_tokens = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            if self.device == "cuda":
                text_tokens = text_tokens.half()
            
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy().flatten()
    
    def extract_features(
        self,
        image: Union[Image.Image, str, Path],
        image_id: Optional[str] = None,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Extract combined image + text features.
        
        Args:
            image: PIL Image, image path, or Path object
            image_id: Unique identifier for caching (optional)
            use_cache: Whether to use cached text descriptions
            
        Returns:
            numpy array: Combined feature vector (512 or 1024 dims)
        """
        # Extract image embedding (always)
        image_emb = self.extract_image_embedding(image)
        
        if not self.use_text:
            return image_emb
        
        # Get text embedding (cached or generate)
        text_emb = None
        if use_cache and image_id:
            text_emb = self.get_cached_text_embedding(image_id)
        
        if text_emb is None:
            # Generate text description
            caption = self.generate_text_description(image)
            if caption:
                text_emb = self.extract_text_embedding(caption)
                # Cache it
                if image_id:
                    self.cache_text_embedding(image_id, text_emb)
            else:
                # Fallback: use empty text embedding
                text_emb = np.zeros(512)
        
        # Concatenate image + text embeddings
        combined = np.concatenate([image_emb, text_emb])
        return combined
    
    def extract_batch(
        self,
        images: list,
        image_ids: Optional[list] = None,
        batch_size: int = 8,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features for multiple images (batched for efficiency).
        
        Args:
            images: List of PIL Images or image paths
            image_ids: List of unique identifiers (optional)
            batch_size: Number of images to process at once
            show_progress: Show progress bar
            
        Returns:
            numpy array: Feature matrix (N x feature_dim)
        """
        features = []
        iterator = tqdm(range(0, len(images), batch_size)) if show_progress else range(0, len(images), batch_size)
        
        for i in iterator:
            batch_images = images[i:i+batch_size]
            batch_ids = image_ids[i:i+batch_size] if image_ids else [None] * len(batch_images)
            
            batch_features = []
            for img, img_id in zip(batch_images, batch_ids):
                feat = self.extract_features(img, img_id, use_cache=True)
                batch_features.append(feat)
            
            features.extend(batch_features)
        
        return np.array(features)


if __name__ == "__main__":
    # Test the extractor
    print("Testing OptimizedImageExtractor...")
    extractor = OptimizedImageExtractor(use_text=False)  # Disable text for quick test
    
    # Test with a dummy image (you'll need a real image path)
    print("\nTo test with a real image:")
    print("  extractor = OptimizedImageExtractor()")
    print("  features = extractor.extract_features('path/to/image.jpg')")
    print(f"  Feature shape: {extractor.extract_features.__annotations__['return']}")

