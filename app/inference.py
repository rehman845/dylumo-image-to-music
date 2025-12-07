"""
Model Inference Module
Handles model loading and prediction
"""
import sys
import json
import time
import torch
import faiss
import pickle
import pandas as pd
from PIL import Image
from pathlib import Path
from transformers import CLIPImageProcessor
import logging

# Add model architecture to path
sys.path.append(str(Path(__file__).parent.parent / 'output' / 'kaggle' / 'working' / 'data'))
from model_architecture import OptimizedCrossModalTransformer

logger = logging.getLogger(__name__)


class ModelInference:
    """Handle model loading and inference"""
    
    def __init__(self, config_module):
        """
        Initialize inference system
        
        Args:
            config_module: Configuration module with paths
        """
        self.config = config_module
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components (loaded lazily)
        self.model = None
        self.processor = None
        self.index = None
        self.metadata = None
        self.deployment_config = None
        
        logger.info(f"Inference system initialized on device: {self.device}")
    
    def load_all(self):
        """Load all model components"""
        logger.info("Loading model components...")
        start_time = time.time()
        
        try:
            # Load deployment config
            self._load_config()
            
            # Load model
            self._load_model()
            
            # Load CLIP processor
            self._load_processor()
            
            # Load FAISS index
            self._load_faiss_index()
            
            # Load metadata
            self._load_metadata()
            
            elapsed = time.time() - start_time
            logger.info(f"✓ All components loaded in {elapsed:.2f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load components: {e}")
            raise
    
    def _load_config(self):
        """Load deployment configuration"""
        logger.info("Loading deployment config...")
        with open(self.config.CONFIG_PATH, 'r') as f:
            self.deployment_config = json.load(f)
        logger.info("✓ Config loaded")
    
    def _load_model(self):
        """Load trained model"""
        logger.info("Loading model...")
        
        # Initialize model
        self.model = OptimizedCrossModalTransformer(
            visual_model=self.deployment_config['visual_model'],
            hidden_dim=self.deployment_config['model_params']['hidden_dim'],
            num_layers=self.deployment_config['model_params']['num_layers'],
            num_heads=self.deployment_config['model_params']['num_heads'],
            dropout=self.deployment_config['model_params']['dropout'],
            audio_feature_dim=len(self.deployment_config['audio_features']),
            num_emotions=len(self.deployment_config['emotions'])
        )
        
        # Load weights
        checkpoint = torch.load(
            self.config.MODEL_PATH,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"✓ Model loaded ({sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters)")
    
    def _load_processor(self):
        """Load CLIP image processor"""
        logger.info("Loading CLIP processor...")
        self.processor = CLIPImageProcessor.from_pretrained(
            self.deployment_config['visual_model']
        )
        logger.info("✓ Processor loaded")
    
    def _load_faiss_index(self):
        """Load FAISS similarity index"""
        logger.info("Loading FAISS index...")
        self.index = faiss.read_index(str(self.config.FAISS_INDEX_PATH))
        logger.info(f"✓ FAISS index loaded ({self.index.ntotal:,} songs)")
    
    def _load_metadata(self):
        """Load song metadata"""
        logger.info("Loading metadata...")
        self.metadata = pd.read_parquet(self.config.METADATA_PATH)
        logger.info(f"✓ Metadata loaded ({len(self.metadata):,} songs)")
    
    def predict(self, image, top_k=10):
        """
        Get song recommendations for an image
        
        Args:
            image: PIL Image
            top_k: Number of recommendations
            
        Returns:
            dict with predictions and recommendations
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load_all() first.")
        
        start_time = time.time()
        
        try:
            # Preprocess image
            pixel_values = self.processor(
                images=image,
                return_tensors='pt'
            )['pixel_values'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(pixel_values, audio_features=None)
                predicted_features = outputs['predicted_features'].cpu().numpy().astype('float32')
                emotion_logits = outputs['emotion_logits'].cpu()
            
            # Get emotion probabilities
            emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
            top_emotion_idx = emotion_probs.argmax().item()
            emotions = self.deployment_config['emotions']
            
            # Normalize features for similarity search
            faiss.normalize_L2(predicted_features)
            
            # Search FAISS index
            scores, ids = self.index.search(predicted_features, top_k)
            
            # Get recommendations
            recommendations = self.metadata.iloc[ids[0]].copy()
            recommendations['similarity_score'] = scores[0]
            recommendations['rank'] = range(1, len(recommendations) + 1)
            
            # Prepare emotion data
            emotion_data = {
                'predicted_emotion': emotions[top_emotion_idx],
                'confidence': float(emotion_probs[top_emotion_idx]),
                'all_probabilities': {
                    emotion: float(prob)
                    for emotion, prob in zip(emotions, emotion_probs)
                }
            }
            
            inference_time = time.time() - start_time
            
            return {
                'status': 'success',
                'emotion': emotion_data,
                'recommendations': recommendations.to_dict('records'),
                'inference_time': f'{inference_time:.2f}s',
                'metadata': {
                    'top_k': top_k,
                    'total_songs_in_db': len(self.metadata),
                    'device': str(self.device)
                }
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def is_loaded(self):
        """Check if all components are loaded"""
        return all([
            self.model is not None,
            self.processor is not None,
            self.index is not None,
            self.metadata is not None
        ])
    
    def get_stats(self):
        """Get system statistics"""
        if not self.is_loaded():
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'device': str(self.device),
            'model_parameters': f"{sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M",
            'songs_indexed': self.index.ntotal,
            'emotions': self.deployment_config['emotions'],
            'visual_model': self.deployment_config['visual_model']
        }

