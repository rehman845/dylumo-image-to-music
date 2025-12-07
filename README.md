# DyLuMo - Image to Music Recommendation System

A cross-modal AI system that recommends music based on image emotions using deep learning.

## Overview

DyLuMo uses a cross-modal transformer architecture combining CLIP vision encoder with custom transformer layers to predict music characteristics from images. The system analyzes image emotions and recommends matching songs from a curated database.

## Architecture

- **Vision Encoder:** CLIP (ViT-Base-Patch32)
- **Cross-Modal Transformer:** 4-layer transformer with 8 attention heads
- **Audio Feature Prediction:** 13 Spotify audio features
- **Emotion Classification:** 7 emotion categories
- **Similarity Search:** FAISS index with 5,782 songs
- **Total Parameters:** 96.5M (9.1M trainable)

## Features

- Image-based music recommendation
- Emotion detection from images
- Fast inference (2 seconds per image)
- Professional dark-themed web interface
- REST API for integration

## Project Structure

```
dylumo-image-to-music/
├── app/                        # Web application (Flask backend + frontend)
│   ├── app.py                  # Flask server
│   ├── inference.py            # Model inference
│   ├── config.py               # Configuration
│   ├── templates/              # HTML templates
│   └── static/                 # JavaScript, CSS
│
├── kaggle/                     # Training notebook
│   └── gen-ai-project-kaggle-merged.ipynb
│
├── output/                     # Trained model artifacts (not in git)
│   └── kaggle/working/
│       ├── checkpoints/        # Model weights
│       └── data/               # FAISS index, metadata
│
├── docs/                       # Project documentation
├── test_inference.py           # Test script
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.8+
- 4GB RAM minimum
- GPU recommended (optional)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd dylumo-image-to-music

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1

# Install dependencies
cd app
pip install -r requirements.txt
```

### Download Model Artifacts

The trained model and FAISS index are not included in git (too large).

**Download from:**
- [Kaggle Output] or [Google Drive] (link after training)
- Extract to `output/kaggle/working/`

**Required files:**
```
output/kaggle/working/
├── checkpoints/dylumo_optimized.pt     (368 MB)
├── data/music_index.index              (300 KB)
├── data/spotify_metadata.parquet       (400 KB)
├── data/spotify_scaler.pkl             (2 KB)
├── data/deployment_config.json         (2 KB)
└── data/model_architecture.py          (5 KB)
```

### Run Application

```bash
# Start Flask server
cd app
python app.py

# Open browser
# Navigate to: http://localhost:5000
```

## Usage

1. **Upload Image:** Click or drag-and-drop an image
2. **Analyze:** Click "Analyze & Recommend"
3. **View Results:** See detected emotion and top 10 song recommendations

## Training

To retrain the model:

1. **Open Kaggle notebook:** `kaggle/gen-ai-project-kaggle-merged.ipynb`
2. **Upload to Kaggle** with GPU enabled
3. **Attach datasets:**
   - `spotify-1million-tracks`
   - `rclone` (for Google Drive sync)
4. **Run all cells** (approximately 3 hours on Tesla T4)
5. **Download artifacts** from `/kaggle/working/`

**Training configuration:**
- Dataset: 25,000 songs, 32,214 images
- Epochs: 30 (20 warmup + 10 fine-tuning)
- Batch size: 128
- Optimizer: AdamW with weight decay
- Regularization: Dropout (0.3), label smoothing (0.2)

## Model Performance

- **Validation Accuracy:** 24-25% (7-class emotion)
- **Test Accuracy:** 25%
- **Feature MSE:** 0.12
- **Baseline (Random):** 14.3%
- **Improvement:** 1.7x over random

**Note:** Accuracy reflects exact emotion match. The system uses predicted audio features for recommendations, which perform better than emotion accuracy alone.

## API Documentation

### Endpoints

**GET /health**
- Check server status and model loading state

**POST /recommend**
- Upload image and get song recommendations
- Request: `multipart/form-data` with `image` field
- Response: JSON with emotion prediction and recommendations

**GET /emotions**
- List all supported emotions

**GET /stats**
- Get system statistics

## Technologies

- **Backend:** Flask, PyTorch, Transformers, FAISS
- **Frontend:** HTML, CSS, JavaScript
- **ML:** CLIP, Custom Transformer, Multi-task Learning
- **Data:** EMID (images), Spotify (music)

## Authors

- 22i-1068
- 22i-0979
- 22i-1243

## License

Academic Project - Fall 2025

## Acknowledgments

- EMID Dataset (ecnu-aigc)
- Spotify Million Tracks Dataset
- OpenAI CLIP Model
- Hugging Face Transformers

## References

- Russell's Circumplex Model of Affect
- CLIP: Learning Transferable Visual Models From Natural Language Supervision
- Cross-Modal Learning for Audio-Visual Recognition
