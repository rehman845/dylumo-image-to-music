# Kaggle Training Setup

This folder contains everything you need to train Dylumo on Kaggle's free GPU.

## Files

- **`kaggle_train.py`** - Complete training script optimized for Kaggle
- **`KAGGLE_SETUP.md`** - Detailed step-by-step guide
- **`notebook_template.ipynb`** - Ready-to-use Kaggle notebook template

## Quick Start

1. Read `KAGGLE_SETUP.md` for complete instructions
2. Prepare your data locally (image + music features)
3. Create a Kaggle dataset and upload your `.npy` files
4. Create a new Kaggle notebook with GPU enabled
5. Copy code from `notebook_template.ipynb` or use `kaggle_train.py`

## Data Requirements

Your Kaggle dataset should contain:
- `image_features.npy` - Image feature matrix (N, 1024)
- `music_features.npy` - Music feature matrix (N, 13)

See `KAGGLE_SETUP.md` for how to prepare these files.

