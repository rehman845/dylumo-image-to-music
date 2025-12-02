# DYLUMO - Image to Music Recommendation

**D**eep **Y**earning **L**earning for **U**nified **M**usic and **O**ptics

A deep learning system that recommends music based on the emotional content of images. Upload an image, get song recommendations that match its mood!

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      TRAINING PHASE (Kaggle GPU)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Images (.jpg/png)  ──►  CLIP ViT-B/32  ──►  Image Features     │
│                                              (512-dim)           │
│                                                   │              │
│                                                   ▼              │
│                                           ┌─────────────┐        │
│  Spotify Dataset    ──►  Normalize +      │  FastMLP    │        │
│  (1M+ songs)            Extract 13        │  Model      │        │
│                         Audio Features    │ 512→256→128 │        │
│                         (13-dim)          │    →13      │        │
│                              ▲            └─────────────┘        │
│                              │                   │               │
│                              └───────────────────┘               │
│                           (matched by emotion)                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image  ──►  CLIP  ──►  FastMLP  ──►  FAISS Search        │
│                    (512-dim)   (13-dim)     (1M+ songs)          │
│                                                   │              │
│                                                   ▼              │
│                                        Top-K Recommended Songs   │
│                                           (with metadata)        │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
dylumo-image-to-music/
├── kaggle/                  # Kaggle training
│   ├── setup_kaggle.py      # Setup Kaggle API credentials
│   ├── train_notebook.ipynb # Training notebook (run on Kaggle GPU)
│   └── kernel-metadata.json # Kaggle API config
├── ml/                      # Core ML modules
│   ├── model.py             # FastMLP architecture
│   ├── extractor.py         # CLIP feature extractor
│   └── recommender.py       # FAISS-based recommender
├── inference/               # Inference scripts (run locally after training)
├── checkpoints/             # Trained models (download from Kaggle)
├── data/                    # Data directory (gitignored)
├── frontend/                # Web frontend (coming soon)
├── backend/                 # API backend (coming soon)
└── docs/                    # Documentation & proposals
```

## Quick Start

### Prerequisites

- Python 3.10+ 
- Git
- Kaggle account (free)

### 1. Clone the Repository

```bash
git clone https://github.com/rehman845/dylumo-image-to-music.git
cd dylumo-image-to-music
```

### 2. Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Kaggle API

```bash
python kaggle/setup_kaggle.py --setup
```

This will ask for your Kaggle credentials. Get them from:
https://www.kaggle.com/settings → API → Create New API Token

---

## Training on Kaggle (Free GPU) 🚀

### Option 1: Upload Notebook Manually (Recommended)

1. Go to https://www.kaggle.com/code
2. Click **"New Notebook"**
3. **File → Import Notebook** → Upload `kaggle/train_notebook.ipynb`
4. Click **"Add Data"** (right sidebar) → Search **"spotify-1million-tracks"** → Add
5. **Settings** → **Accelerator** → Select **"GPU T4 x2"**
6. Click **"Run All"** and wait (~20-30 mins)

### Option 2: Push via Kaggle API

```bash
python kaggle/setup_kaggle.py --push
```

### After Training Completes:

Download these files from Kaggle **Output** tab:
- `dylumo_model.pt`
- `spotify_scaler.pkl` 
- `spotify_features.npy`
- `spotify_metadata.parquet`

Place them in your local folders:
```
checkpoints/
├── dylumo_model.pt
└── spotify_scaler.pkl

data/processed/
├── spotify_features.npy
└── spotify_metadata.parquet
```

---

## Inference (After Training)

### 5. Build FAISS Index

```bash
python inference/build_index.py
```

### 6. Get Recommendations

```bash
python inference/recommend.py --image path/to/your/image.jpg --top_k 10
```

---

## Datasets

### Spotify 1M Tracks
- **Source:** [Kaggle](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks)
- **Size:** 1.16M tracks
- **Features:** 13 audio features (danceability, energy, valence, tempo, etc.)
- **Note:** Added automatically when you add it to Kaggle notebook

### EMID Dataset
- **Source:** [HuggingFace](https://huggingface.co/datasets/ecnu-aigc/EMID)
- **Paper:** [EMID: An Emotional Aligned Dataset](https://arxiv.org/abs/2308.07622)
- **Size:** 10,738 music-image pairs, 3,240 unique images
- **Emotions:** anger, amusement, fear, sadness, excitement, awe, contentment
- **Note:** Downloaded automatically in the notebook

---

## Model Architecture

**FastMLP** - Maps image features to music features:

```
Input (512-dim CLIP features)
    │
    ▼
Linear(512, 512) + BatchNorm + ReLU + Dropout(0.2)
    │
    ▼
Linear(512, 256) + BatchNorm + ReLU + Dropout(0.2)
    │
    ▼
Linear(256, 128) + BatchNorm + ReLU + Dropout(0.2)
    │
    ▼
Linear(128, 13)
    │
    ▼
Output (13-dim audio features)
```

**Parameters:** ~430,000

---

## Audio Features (13 dimensions)

| Feature | Description | Range |
|---------|-------------|-------|
| danceability | How suitable for dancing | 0-1 |
| energy | Perceptual intensity | 0-1 |
| key | Musical key | 0-11 (normalized) |
| loudness | Overall loudness (dB) | normalized |
| mode | Major (1) or minor (0) | 0-1 |
| speechiness | Presence of spoken words | 0-1 |
| acousticness | Acoustic confidence | 0-1 |
| instrumentalness | No vocals prediction | 0-1 |
| liveness | Audience presence | 0-1 |
| valence | Musical positiveness | 0-1 |
| tempo | BPM | normalized |
| duration_ms | Track length | normalized |
| time_signature | Beats per bar | normalized |

---

## Emotion Mapping

Images and songs are matched by emotion categories:

| Valence | Energy | Emotion |
|---------|--------|---------|
| High | High | excitement |
| High | Medium | amusement |
| High | Low | contentment |
| Low | High | anger |
| Low | Medium | fear |
| Low | Low | sadness |

---

## For Team Members

### Complete Setup (Run in Order)

```bash
# ============================================
# STEP 1: Clone & Setup Environment
# ============================================
git clone https://github.com/rehman845/dylumo-image-to-music.git
cd dylumo-image-to-music

# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac (use this instead)
# python -m venv venv
# source venv/bin/activate

pip install -r requirements.txt

# ============================================
# STEP 2: Setup Kaggle API
# ============================================
python kaggle/setup_kaggle.py --setup
# Enter your Kaggle username and API key when prompted
# Get credentials from: https://www.kaggle.com/settings → API

# ============================================
# STEP 3: Train on Kaggle (Manual - Recommended)
# ============================================
# 1. Go to https://www.kaggle.com/code
# 2. Click "New Notebook"
# 3. File → Import Notebook → Upload kaggle/train_notebook.ipynb
# 4. Add Data → Search "spotify-1million-tracks" → Add
# 5. Settings → Accelerator → GPU T4 x2
# 6. Run All (wait ~20-30 mins)
# 7. Download output files when complete

# ============================================
# STEP 4: Download Kaggle Outputs
# ============================================
# Create folders if they don't exist
mkdir -p checkpoints data/processed

# Download from Kaggle Output tab and place:
# - dylumo_model.pt → checkpoints/
# - spotify_scaler.pkl → checkpoints/
# - spotify_features.npy → data/processed/
# - spotify_metadata.parquet → data/processed/

# ============================================
# STEP 5: Build FAISS Index
# ============================================
python inference/build_index.py

# ============================================
# STEP 6: Test Recommendations
# ============================================
python inference/recommend.py --image path/to/image.jpg --top_k 10
```

---

## Team

- Adeel Mahmood Ansari (22i-0979)
- Awab (22i-1068)
- Talha Azim (22i-1243)

## License

This project is for educational purposes (Generative AI Course - Fall 2025).

## Acknowledgments

- [EMID Dataset](https://arxiv.org/abs/2308.07622) - ECNU AIGC Lab
- [Spotify 1M Tracks](https://www.kaggle.com/datasets/amitanshjoshi/spotify-1million-tracks) - Amitansh Joshi
- [OpenAI CLIP](https://github.com/openai/CLIP)
