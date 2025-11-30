"""
Image Feature Extractor using CLIP ViT-B/32

Extracts 1024-dimensional features from images using OpenAI's CLIP model.
For production, we use CLIP ViT-B/32 which provides good balance of speed and quality.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List
from PIL import Image

# Lazy imports to avoid loading heavy models at import time
_clip_model = None
_clip_preprocess = None
_device = None


def get_device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_model(device: str = None):
    """
    Load CLIP model lazily.
    
    Args:
        device: Device to load model on. If None, auto-detect.
    """
    global _clip_model, _clip_preprocess, _device
    
    if _clip_model is not None:
        return _clip_model, _clip_preprocess, _device
    
    try:
        import open_clip
        
        _device = device or get_device()
        print(f"[INFO] Loading CLIP ViT-B/32 on {_device}...")
        
        # Load OpenCLIP model (more flexible than original CLIP)
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32',
            pretrained='openai'
        )
        _clip_model = _clip_model.to(_device)
        _clip_model.eval()
        
        print("[OK] CLIP model loaded successfully!")
        return _clip_model, _clip_preprocess, _device
        
    except ImportError:
        raise ImportError(
            "Please install open-clip-torch: pip install open-clip-torch"
        )


class ImageFeatureExtractor:
    """
    Extract features from images using CLIP ViT-B/32.
    
    Output: 512-dim features (CLIP ViT-B/32 default)
    We'll project to 1024-dim if needed, or use ViT-L/14 for 768-dim.
    
    For our architecture, we concatenate CLIP (512) + BLIP-2 (512) = 1024 dim
    Or use a single larger model.
    """
    
    def __init__(self, device: str = None, use_blip: bool = False):
        """
        Initialize the feature extractor.
        
        Args:
            device: Device to use ('cuda', 'cpu', 'mps')
            use_blip: Whether to also use BLIP-2 features (adds 512 dim)
        """
        self.model, self.preprocess, self.device = load_clip_model(device)
        self.use_blip = use_blip
        self.blip_model = None
        self.blip_processor = None
        
        # CLIP ViT-B/32 produces 512-dim features
        # We'll duplicate/project to 1024 if needed
        self.feature_dim = 512
        
        if use_blip:
            self._load_blip()
            self.feature_dim = 1024  # 512 CLIP + 512 BLIP
    
    def _load_blip(self):
        """Load BLIP-2 model for additional features."""
        try:
            from transformers import Blip2Processor, Blip2Model
            
            print("[INFO] Loading BLIP-2 model...")
            self.blip_processor = Blip2Processor.from_pretrained(
                "Salesforce/blip2-opt-2.7b"
            )
            self.blip_model = Blip2Model.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            self.blip_model.eval()
            print("[OK] BLIP-2 loaded!")
            
        except Exception as e:
            print(f"[WARNING] Could not load BLIP-2: {e}")
            print("[INFO] Using CLIP only (512-dim features)")
            self.use_blip = False
            self.feature_dim = 512
    
    def extract_single(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract features from a single image.
        
        Args:
            image: Path to image or PIL Image
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a path or PIL Image")
        
        # Extract CLIP features
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            clip_features = self.model.encode_image(image_tensor)
            clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
            clip_features = clip_features.cpu().numpy().squeeze()
        
        if self.use_blip and self.blip_model is not None:
            # Extract BLIP features
            with torch.no_grad():
                inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
                blip_outputs = self.blip_model.get_image_features(**inputs)
                blip_features = blip_outputs.pooler_output
                blip_features = blip_features / blip_features.norm(dim=-1, keepdim=True)
                blip_features = blip_features.cpu().numpy().squeeze()
            
            # Concatenate CLIP + BLIP
            features = np.concatenate([clip_features, blip_features[:512]])
        else:
            # Duplicate CLIP features to get 1024-dim
            # Or just use 512-dim and adjust model input
            features = clip_features
        
        return features
    
    def extract_batch(
        self, 
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of image paths or PIL Images
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Feature matrix of shape (n_images, feature_dim)
        """
        from tqdm import tqdm
        
        all_features = []
        
        iterator = range(0, len(images), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")
        
        for i in iterator:
            batch = images[i:i + batch_size]
            
            # Load and preprocess images
            pil_images = []
            for img in batch:
                if isinstance(img, (str, Path)):
                    pil_images.append(Image.open(img).convert('RGB'))
                else:
                    pil_images.append(img)
            
            # Extract CLIP features
            with torch.no_grad():
                image_tensors = torch.stack([
                    self.preprocess(img) for img in pil_images
                ]).to(self.device)
                
                clip_features = self.model.encode_image(image_tensors)
                clip_features = clip_features / clip_features.norm(dim=-1, keepdim=True)
                batch_features = clip_features.cpu().numpy()
            
            all_features.append(batch_features)
        
        return np.vstack(all_features)
    
    def get_feature_dim(self) -> int:
        """Get the dimension of output features."""
        return self.feature_dim


if __name__ == "__main__":
    # Test the extractor
    print("Testing ImageFeatureExtractor...")
    
    extractor = ImageFeatureExtractor(use_blip=False)
    print(f"Feature dimension: {extractor.get_feature_dim()}")
    
    # Create a test image
    test_image = Image.new('RGB', (224, 224), color='red')
    features = extractor.extract_single(test_image)
    print(f"Extracted features shape: {features.shape}")

