# Quick Start Guide - Dylumo

## 🚀 Complete Pipeline Overview

Your project now has all the core components:

1. **Image Feature Extractor** (`src/feature_extractors/optimized_image_extractor.py`)
   - CLIP ViT-B/32 for fast image embeddings
   - Optional BLIP-2 for text descriptions (cached)

2. **Fast MLP Model** (`src/models/fast_mlp.py`)
   - 3-layer MLP optimized for speed + accuracy
   - Maps image features → music features (13 dimensions)

3. **FAISS Recommender** (`src/recommender/faiss_recommender.py`)
   - GPU-accelerated similarity search
   - Instant recommendations from 100K+ songs

4. **Data Preparation** (`src/data/prepare_data.py`)
   - Loads Spotify 100K subset
   - Creates training pairs

5. **Training Script** (`src/train.py`)
   - Complete training pipeline with validation

---

## 📋 Step-by-Step Workflow

### Step 1: Install Dependencies

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Mac/Linux

# Install all packages
pip install -r requirements.txt

# For GPU support (if you have CUDA):
# pip install faiss-gpu  # Instead of faiss-cpu
```

### Step 2: Prepare Your Data

#### 2a. Download Spotify Dataset
- Go to Kaggle: "1 Million Spotify Tracks"
- Download the CSV file
- Place it in `data/raw/spotify_tracks.csv`

#### 2b. Prepare Music Data (100K subset)

```python
from src.data.prepare_data import prepare_training_data

# This will:
# - Load CSV
# - Sample 100K songs
# - Extract and normalize features
# - Save to data/processed/
data = prepare_training_data(
    spotify_csv='data/raw/spotify_tracks.csv',
    n_songs=100000,
    output_dir='data/processed'
)
```

#### 2c. Build FAISS Index

```python
from src.recommender.faiss_recommender import FAISSRecommender
import pandas as pd
import numpy as np

# Load music features
music_features = np.load('data/processed/music_features.npy')
music_metadata = pd.read_csv('data/processed/music_metadata.csv')

# Create and save index
recommender = FAISSRecommender(
    music_features=music_features,
    music_metadata=music_metadata,
    use_gpu=True  # Use GPU if available
)

recommender.save_index('data/processed/music_index.faiss')
```

### Step 3: Prepare Training Images

You need images with corresponding music features. Options:

#### Option A: Use Existing Image-Music Datasets
- Music4All, MusicCaps, or similar
- Extract image features for all images

#### Option B: Create Your Own Pairs
- Collect images (emotion-labeled datasets like AffectNet)
- Manually or semi-automatically match to songs

#### Extract Image Features

```python
from src.feature_extractors.optimized_image_extractor import OptimizedImageExtractor
from pathlib import Path
import numpy as np

extractor = OptimizedImageExtractor(use_text=True)  # Enable text for better results

# Process all training images
image_dir = Path('data/images/train')
image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

image_features = []
for img_path in image_paths:
    feat = extractor.extract_features(img_path, image_id=img_path.stem)
    image_features.append(feat)

image_features = np.array(image_features)
np.save('data/processed/image_features.npy', image_features)
```

### Step 4: Create Training Pairs

```python
from src.data.prepare_data import create_training_pairs
import numpy as np

# Load features
image_features = np.load('data/processed/image_features.npy')
music_features = np.load('data/processed/music_features.npy')

# Create pairs (random or similarity-based)
img_pairs, music_pairs, pair_ids = create_training_pairs(
    image_features=image_features,
    image_ids=[f'img_{i}' for i in range(len(image_features))],
    music_features=music_features,
    music_ids=[f'song_{i}' for i in range(len(music_features))],
    n_pairs=50000,  # 50K training pairs
    strategy='random'
)

# Save pairs
np.save('data/processed/train_image_features.npy', img_pairs)
np.save('data/processed/train_music_features.npy', music_pairs)
```

### Step 5: Train Model

```bash
python src/train.py \
    --image-features data/processed/train_image_features.npy \
    --music-features data/processed/train_music_features.npy \
    --output-dir models \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --input-dim 1024
```

This will:
- Train for 50 epochs
- Save best model to `models/best_model.pth`
- Show training/validation loss

### Step 6: Test End-to-End

```bash
python src/example_end_to_end.py \
    --image path/to/test/image.jpg \
    --model models/best_model.pth \
    --music-index data/processed/music_index.faiss \
    --k 10 \
    --use-text
```

---

## 🎯 Expected Performance

- **Image Feature Extraction**: ~100-200ms per image
- **Model Prediction**: ~5ms per image
- **Recommendation Search**: <1ms (FAISS GPU)
- **Total Pipeline**: ~200ms end-to-end

---

## 📊 Next Steps

1. **Collect Training Images**: Get image-music pairs
2. **Extract Features**: Run feature extraction on all images
3. **Train Model**: Run training script
4. **Evaluate**: Test on validation set
5. **Deploy**: Build API or web interface

---

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce batch size: `--batch-size 32`
- Use CPU: `--device cpu`

### "FAISS not found"
- Install: `pip install faiss-cpu` (or `faiss-gpu`)

### "CLIP model not loading"
- Check internet connection (downloads model first time)
- Or download manually and set `CLIP_MODEL_PATH`

---

## 📝 Notes

- **100K songs** is a good balance: enough for training, fast enough to process
- **Text descriptions** improve accuracy but slow down feature extraction (cache them!)
- **GPU recommended** for training and FAISS search (10x faster)
- **Mixed precision (FP16)** gives 2x speedup with minimal accuracy loss

---

## 🎉 You're Ready!

All the code is in place. Now you just need:
1. Your Spotify dataset CSV
2. Training images (with music matches)
3. Run the scripts above

Good luck! 🚀

