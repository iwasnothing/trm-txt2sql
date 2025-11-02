#!/usr/bin/env python3
"""
Download and extract Spider SQL dataset
"""
import os
import urllib.request
import urllib.error
import zipfile
import json
from pathlib import Path


class DownloadProgressBar:
    """Progress bar for file downloads"""
    def __init__(self):
        self.bytes_downloaded = 0
        self.total_size = 0
    
    def __call__(self, block_num, block_size, total_size):
        if total_size > 0:
            self.total_size = total_size
            self.bytes_downloaded = block_num * block_size
            percent = min(100, (self.bytes_downloaded / total_size) * 100)
            if block_num % 10 == 0:  # Update every 10 blocks to avoid spam
                print(f"\r  Progress: {percent:.1f}% ({self.bytes_downloaded // 1024 // 1024}MB/{total_size // 1024 // 1024}MB)", end='', flush=True)


def download_file(url: str, filepath: Path, desc: str = None):
    """Download a file from URL with progress indication"""
    try:
        print(f"Downloading {desc or filepath.name}...")
        # Create request with User-Agent header to avoid 403 errors
        req = urllib.request.Request(url)
        req.add_header('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Use urlopen and save manually to support Request object
        with urllib.request.urlopen(req) as response:
            total_size = int(response.headers.get('Content-Length', 0))
            
            with open(filepath, 'wb') as f:
                downloaded = 0
                block_size = 8192
                
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0 and downloaded % (block_size * 10) == 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded // 1024 // 1024
                        mb_total = total_size // 1024 // 1024
                        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded}MB/{mb_total}MB)", end='', flush=True)
        
        print(f"\r  ✓ Saved to {filepath}")
        return True
    except urllib.error.HTTPError as e:
        print(f"\n  ✗ HTTP Error {e.code} downloading {desc or filepath.name}: {e.reason}")
        print(f"     URL: {url}")
        return False
    except Exception as e:
        print(f"\n  ✗ Error downloading {desc or filepath.name}: {e}")
        return False


def download_spider_dataset(output_dir="data"):
    """
    Download Spider dataset from GitHub repository
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Spider dataset URLs from GitHub
    # Files are located in evaluation_examples/examples/ directory
    base_url = "https://raw.githubusercontent.com/taoyds/spider/master/evaluation_examples/examples/"
    
    # Required files for training
    required_files = {
        "train_spider.json": base_url + "train_spider.json",
        "dev.json": base_url + "dev.json",
        "tables.json": base_url + "tables.json"
    }
    
    # Optional files (may not exist)
    optional_files = {
        "train_others.json": base_url + "train_others.json"
    }
    
    # Create directory structure
    spider_dir = output_path / "spider"
    spider_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Downloading Spider Dataset from GitHub")
    print("=" * 60)
    print(f"Output directory: {spider_dir}\n")
    
    # Check which files are missing
    files_to_download = {}
    
    # Check required files
    for filename, url in required_files.items():
        filepath = spider_dir / filename
        if not filepath.exists():
            files_to_download[filename] = (url, filepath)
        else:
            print(f"  ⊗ {filename} already exists, skipping...")
    
    # Also check optional files
    for filename, url in optional_files.items():
        filepath = spider_dir / filename
        if not filepath.exists():
            files_to_download[filename] = (url, filepath)
    
    if not files_to_download:
        print("\n✓ All required files already exist!")
        return True
    
    # Separate required and optional files
    required_to_download = {k: v for k, v in files_to_download.items() 
                            if k in required_files}
    optional_to_download = {k: v for k, v in files_to_download.items() 
                           if k in optional_files}
    
    # Download required files
    if required_to_download:
        print(f"\nDownloading {len(required_to_download)} required file(s)...\n")
        required_success = 0
        for filename, (url, filepath) in required_to_download.items():
            if download_file(url, filepath, filename):
                required_success += 1
        
        if required_success < len(required_to_download):
            print("\n" + "=" * 60)
            print(f"✗ Failed to download {len(required_to_download) - required_success} required file(s)")
            print("Please download manually from: https://github.com/taoyds/spider")
            return False
    
    # Download optional files
    if optional_to_download:
        print(f"\nDownloading {len(optional_to_download)} optional file(s)...\n")
        for filename, (url, filepath) in optional_to_download.items():
            download_file(url, filepath, filename)
            # Don't fail if optional files fail
    
    print("\n" + "=" * 60)
    required_downloaded = len([f for f in required_files.keys() 
                              if (spider_dir / f).exists()])
    if required_downloaded == len(required_files):
        print(f"✓ Successfully downloaded all required files ({required_downloaded}/{len(required_files)})")
        return True
    else:
        print(f"⚠ Only {required_downloaded}/{len(required_files)} required files are available")
        return False


def verify_dataset(data_dir="data/spider"):
    """Verify that required dataset files exist"""
    required = ["train_spider.json", "dev.json", "tables.json"]
    data_path = Path(data_dir)
    
    for filename in required:
        filepath = data_path / filename
        if not filepath.exists():
            print(f"Missing: {filepath}")
            return False
    
    # Check file contents
    try:
        with open(data_path / "train_spider.json") as f:
            train_data = json.load(f)
            print(f"Train examples: {len(train_data)}")
        
        with open(data_path / "dev.json") as f:
            dev_data = json.load(f)
            print(f"Dev examples: {len(dev_data)}")
        
        with open(data_path / "tables.json") as f:
            tables_data = json.load(f)
            print(f"Database schemas: {len(tables_data)}")
        
        return True
    except Exception as e:
        print(f"Error verifying dataset: {e}")
        return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Spider dataset")
    parser.add_argument("--data_dir", default="data", help="Output directory")
    args = parser.parse_args()
    
    if download_spider_dataset(args.data_dir):
        verify_dataset(f"{args.data_dir}/spider")

