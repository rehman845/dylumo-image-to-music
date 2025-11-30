# Training Dylumo on Kaggle

Complete guide to train your model using Kaggle's free GPU resources.

## 🚀 Quick Start

1. **Prepare your data** (see below)
2. **Create a Kaggle Dataset** with your prepared features
3. **Create a new Kaggle Notebook**
4. **Copy the training code** from `kaggle_train.py`
5. **Run with GPU enabled**

---

## 📦 Step 1: Prepare Data Locally

Before uploading to Kaggle, you need to prepare your data:

### 1a. Extract Image Features

```python
# Run this locally first
from src.feature_extractors.optimized_image_extractor import OptimizedImageExtractor
import numpy as np
from pathlib import Path

extractor = OptimizedImageExtractor(use_text=True)

# Process all your training images
image_dir = Path('data/images/train')
image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))

image_features = []
for img_path in image_paths:
    feat = extractor.extract_features(img_path, image_id=img_path.stem)
    image_features.append(feat)

image_features = np.array(image_features)
np.save('data/processed/image_features.npy', image_features)
```

### 1b. Prepare Music Features

```python
from src.data.prepare_data import prepare_training_data

# This creates music_features.npy (100K songs)
data = prepare_training_data(
    spotify_csv='data/raw/spotify_tracks.csv',
    n_songs=100000,
    output_dir='data/processed'
)
```

### 1c. Create Training Pairs

```python
from src.data.prepare_data import create_training_pairs
import numpy as np

image_features = np.load('data/processed/image_features.npy')
music_features = np.load('data/processed/music_features.npy')

# Create pairs
img_pairs, music_pairs, _ = create_training_pairs(
    image_features=image_features,
    image_ids=[f'img_{i}' for i in range(len(image_features))],
    music_features=music_features,
    music_ids=[f'song_{i}' for i in range(len(music_features))],
    n_pairs=50000,  # Adjust based on your data
    strategy='random'
)

# Save for Kaggle
np.save('data/processed/train_image_features.npy', img_pairs)
np.save('data/processed/train_music_features.npy', music_pairs)
```

---

## 📤 Step 2: Create Kaggle Dataset

1. **Go to Kaggle**: https://www.kaggle.com/datasets
2. **Click "New Dataset"**
3. **Upload your files**:
   - `train_image_features.npy`
   - `train_music_features.npy`
   - `src/` folder (your code)
4. **Name it**: `dylumo-data` (or any name you prefer)
5. **Make it public or private** (private is fine, you can access it)

**Important**: The dataset structure should be:
```
dylumo-data/
├── image_features.npy
├── music_features.npy
└── src/  (optional, if you want to include code)
```

---

## 💻 Step 3: Create Kaggle Notebook

1. **Go to**: https://www.kaggle.com/code
2. **Click "New Notebook"**
3. **Settings**:
   - **Accelerator**: GPU (P100 or T4)
   - **Internet**: ON (to download models)
   - **Language**: Python

---

## 📝 Step 4: Setup Notebook Code

### Cell 1: Install Dependencies

```python
!pip install faiss-cpu  # or faiss-gpu if available
!pip install clip-by-openai
```

### Cell 2: Copy Your Code

Copy the entire `kaggle_train.py` file content into a code cell, OR:

```python
# Option A: If you included src/ in your dataset
import sys
sys.path.append('/kaggle/input/dylumo-data/src')

from models.fast_mlp import FastMLP, WeightedMSELoss
# ... rest of training code
```

### Cell 3: Run Training

```python
# The main() function will run automatically
# Or call it explicitly:
main()
```

---

## 🔧 Step 5: Configure Paths

In the notebook, make sure paths match your dataset name:

```python
# If your dataset is named "dylumo-data"
INPUT_DIR = Path('/kaggle/input/dylumo-data')

# If your dataset has a different name, update:
config = {
    'image_features_path': INPUT_DIR / 'image_features.npy',
    'music_features_path': INPUT_DIR / 'music_features.npy',
    ...
}
```

---

## ⚙️ Kaggle-Specific Optimizations

### GPU Settings
- Kaggle provides **P100 or T4 GPUs** (free tier)
- **30 hours/week** of GPU time
- **Batch size**: Use 128-256 (larger than local training)

### Memory Management
```python
# If you run out of memory, reduce batch size:
config['batch_size'] = 64  # or 32

# Or use gradient accumulation:
# (accumulate gradients over multiple batches)
```

### Save Outputs
```python
# Kaggle saves /kaggle/working/ automatically
# Your trained models will be in the output
```

---

## 📊 Monitoring Training

### View Training Progress
- Kaggle shows output in real-time
- Check loss values in notebook output
- Models are saved to `/kaggle/working/models/`

### Download Results
1. After training completes
2. Go to notebook **Output** tab
3. Download `best_model.pth` and `final_model.pth`

---

## 🐛 Troubleshooting

### "Module not found"
```python
# Make sure you added src/ to your dataset
# Or install packages:
!pip install <package-name>
```

### "CUDA out of memory"
```python
# Reduce batch size:
config['batch_size'] = 32

# Or use gradient checkpointing
```

### "Dataset not found"
- Check dataset name matches in code
- Make sure dataset is attached to notebook
- Click "Add Data" → Search for your dataset

---

## 📈 Expected Performance on Kaggle

- **Training time**: ~10-30 minutes for 50 epochs (depending on data size)
- **GPU utilization**: Should be 80-100%
- **Memory usage**: ~2-4 GB GPU memory

---

## 🎯 Complete Example Notebook Structure

```python
# Cell 1: Setup
!pip install faiss-cpu clip-by-openai

# Cell 2: Imports
import torch
import numpy as np
from pathlib import Path
# ... all imports

# Cell 3: Load code (if in dataset)
import sys
sys.path.append('/kaggle/input/dylumo-data/src')
from models.fast_mlp import FastMLP, WeightedMSELoss

# Cell 4: Training code
# (Copy kaggle_train.py content here)

# Cell 5: Run
main()
```

---

## ✅ Checklist

- [ ] Data prepared locally (image + music features)
- [ ] Kaggle dataset created and uploaded
- [ ] Notebook created with GPU enabled
- [ ] Code copied to notebook
- [ ] Paths configured correctly
- [ ] Training started
- [ ] Models downloaded after completion

---

## 🚀 You're Ready!

Follow these steps and you'll be training on Kaggle's free GPU in no time!

**Pro Tip**: Save your notebook as a template so you can reuse it for future training runs.

