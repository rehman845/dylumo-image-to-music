"""
Configuration for Flask Backend
Paths to model artifacts and settings
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'output' / 'kaggle' / 'working'
DATA_DIR = OUTPUT_DIR / 'data'
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'

# Model artifacts paths
MODEL_PATH = CHECKPOINT_DIR / 'dylumo_optimized.pt'
FAISS_INDEX_PATH = DATA_DIR / 'music_index.index'
METADATA_PATH = DATA_DIR / 'spotify_metadata.parquet'
SCALER_PATH = DATA_DIR / 'spotify_scaler.pkl'
CONFIG_PATH = DATA_DIR / 'deployment_config.json'
MODEL_ARCHITECTURE_PATH = DATA_DIR / 'model_architecture.py'

# Model parameters (will be loaded from config)
VISUAL_MODEL = 'openai/clip-vit-base-patch32'
HIDDEN_DIM = 512
NUM_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.3

# API settings
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif'}
DEFAULT_TOP_K = 10
MAX_TOP_K = 50

# Server settings
HOST = '0.0.0.0'
PORT = 5000
DEBUG = True

