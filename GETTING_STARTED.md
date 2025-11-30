# Getting Started with Dylumo

## Step 1: Set Up Your Environment (5 minutes)

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate it:**
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Test That Everything Works (2 minutes)

Run the basic example:
```bash
python src/example_basic.py
```

You should see:
```
Dylumo - Basic Example
==================================================
Loading CLIP model...
✓ CLIP loaded successfully on cpu
  Model: ViT-B/32
  Image embedding dimension: 512
```

**If you see this, you're ready to go!** ✅

## Step 3: Your First Real Task - Load an Image (10 minutes)

1. **Find any image** (jpg, png) on your computer
2. **Copy it to:** `data/test_images/` (create the folder if needed)
3. **Modify `src/example_basic.py`** to accept an image path:
   ```python
   if __name__ == "__main__":
       import sys
       if len(sys.argv) > 1:
           image_path = sys.argv[1]
           features = load_image_features(image_path)
           print(f"Image features shape: {features.shape}")
           print(f"First 5 values: {features[:5]}")
       else:
           # ... existing code ...
   ```
4. **Run it:**
   ```bash
   python src/example_basic.py data/test_images/your_image.jpg
   ```

## Step 4: Next Steps (What to Do After This Works)

### Immediate Next Tasks:
1. **Set up Spotify API** (for music data):
   - Go to https://developer.spotify.com/dashboard
   - Create an app
   - Get Client ID and Secret
   - Add them to `.env` file (copy from `.env.example`)

2. **Create a simple music loader:**
   - Create `src/load_music.py`
   - Use spotipy to fetch a few songs
   - Extract audio features

3. **Build your first model:**
   - Create `src/models/baseline.py`
   - Simple cosine similarity between image and music embeddings

### Week 1 Goals:
- [ ] Environment set up and working
- [ ] Can load images and extract features
- [ ] Can load music data from Spotify
- [ ] Have a simple similarity function working

## Need Help?

If something doesn't work:
1. Check error messages carefully
2. Make sure all dependencies installed: `pip install -r requirements.txt`
3. Check Python version: `python --version` (should be 3.11+)

## What Each Folder Is For:

- `src/` - Your main Python code
- `notebooks/` - Jupyter notebooks for experiments
- `data/` - Your datasets (images, music files)
- `experiments/` - Results, plots, tables
- `docker/` - Docker configuration
- `paper/` - Your research paper (LaTeX)
- `prompts/` - Log of all GPT prompts you use

