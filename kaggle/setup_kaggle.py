"""
DYLUMO - Kaggle Setup Script

This script sets up Kaggle API credentials for:
1. Pushing training notebooks to Kaggle
2. Downloading trained models from Kaggle

Usage:
    python kaggle/setup_kaggle.py --setup      # Setup Kaggle credentials
    python kaggle/setup_kaggle.py --push       # Push notebook to Kaggle
    python kaggle/setup_kaggle.py --status     # Check notebook status
"""

import os
import sys
import json
import argparse
from pathlib import Path


def get_kaggle_dir():
    """Get the Kaggle configuration directory based on OS."""
    return Path.home() / ".kaggle"


def setup_kaggle_credentials():
    """Interactive setup for Kaggle API credentials."""
    print("\n" + "=" * 50)
    print("[KEY] Kaggle API Setup")
    print("=" * 50)
    print("\nTo get your Kaggle API credentials:")
    print("1. Go to https://www.kaggle.com/settings")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads kaggle.json with your credentials\n")
    
    kaggle_dir = get_kaggle_dir()
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Check if already configured
    if kaggle_json.exists():
        print(f"[OK] Kaggle credentials already exist at: {kaggle_json}")
        overwrite = input("Do you want to overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Keeping existing credentials.")
            return True
    
    # Get credentials from user
    print("\nEnter your Kaggle credentials:")
    username = input("Kaggle Username: ").strip()
    api_key = input("Kaggle API Key: ").strip()
    
    if not username or not api_key:
        print("[ERROR] Username and API key are required!")
        return False
    
    # Create kaggle directory if it doesn't exist
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write credentials
    credentials = {"username": username, "key": api_key}
    
    with open(kaggle_json, 'w') as f:
        json.dump(credentials, f)
    
    # Set permissions (Unix only)
    if sys.platform != "win32":
        os.chmod(kaggle_json, 0o600)
    
    print(f"\n[OK] Kaggle credentials saved to: {kaggle_json}")
    return True


def verify_kaggle_setup():
    """Verify Kaggle API is properly configured."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("[OK] Kaggle API authenticated successfully!")
        return api
    except Exception as e:
        print(f"[ERROR] Kaggle API authentication failed: {e}")
        return None


def push_notebook():
    """Push the training notebook to Kaggle."""
    api = verify_kaggle_setup()
    if not api:
        return False
    
    kaggle_dir = Path(__file__).parent
    
    print("\n[INFO] Pushing notebook to Kaggle...")
    print(f"   Notebook: {kaggle_dir / 'train_notebook.ipynb'}")
    print(f"   Metadata: {kaggle_dir / 'kernel-metadata.json'}")
    
    try:
        api.kernels_push(str(kaggle_dir))
        print("[OK] Notebook pushed successfully!")
        print("\nView your notebook at:")
        print("   https://www.kaggle.com/code")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to push notebook: {e}")
        return False


def check_status():
    """Check the status of the running notebook."""
    api = verify_kaggle_setup()
    if not api:
        return False
    
    # Read kernel metadata to get the kernel name
    kaggle_dir = Path(__file__).parent
    metadata_file = kaggle_dir / "kernel-metadata.json"
    
    if not metadata_file.exists():
        print("[ERROR] kernel-metadata.json not found")
        return False
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    kernel_id = metadata.get("id", "")
    
    print(f"\n[INFO] Checking status of: {kernel_id}")
    
    try:
        status = api.kernels_status(kernel_id)
        print(f"[OK] Status: {status}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to check status: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="DYLUMO - Kaggle Setup Script")
    parser.add_argument("--setup", action="store_true", help="Setup Kaggle credentials")
    parser.add_argument("--push", action="store_true", help="Push notebook to Kaggle")
    parser.add_argument("--status", action="store_true", help="Check notebook status")
    
    args = parser.parse_args()
    
    # Default to --setup if no arguments provided
    if not (args.setup or args.push or args.status):
        args.setup = True
    
    print("\n" + "=" * 60)
    print("DYLUMO - Image to Music Recommendation")
    print("Kaggle Setup Script")
    print("=" * 60)
    
    if args.setup:
        if not setup_kaggle_credentials():
            print("\n[ERROR] Setup failed. Please try again.")
            return
        verify_kaggle_setup()
    
    if args.push:
        push_notebook()
    
    if args.status:
        check_status()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("=" * 60)
    print("\n1. Push notebook to Kaggle:")
    print("   python kaggle/setup_kaggle.py --push")
    print("\n2. Or manually upload:")
    print("   - Go to https://www.kaggle.com/code")
    print("   - Upload kaggle/train_notebook.ipynb")
    print("   - Add dataset: spotify-1million-tracks")
    print("   - Enable GPU and run!")
    print()


if __name__ == "__main__":
    main()
