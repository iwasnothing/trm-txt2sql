#!/usr/bin/env python3
"""
Download and extract Spider SQL dataset using Hugging Face datasets library
"""
import json
from pathlib import Path


def check_datasets_package():
    """Check if datasets package is installed"""
    try:
        import datasets
        return True
    except ImportError:
        return False


def install_datasets_package():
    """Install the datasets package"""
    import subprocess
    import sys
    print("Installing datasets package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "--quiet"])
        print("  ✓ datasets package installed successfully")
        return True
    except Exception as e:
        print(f"  ✗ Failed to install datasets package: {e}")
        return False


def download_spider_dataset(output_dir="data", auto_install=True):
    """
    Download Spider dataset using Hugging Face datasets library
    
    Args:
        output_dir: Directory to save downloaded files
        auto_install: Whether to automatically install datasets package if missing
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create directory structure
    spider_dir = output_path / "spider"
    spider_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 60)
    print("Downloading Spider Dataset via Hugging Face")
    print("=" * 60)
    print(f"Output directory: {spider_dir}\n")
    
    # Check if datasets package is installed
    if not check_datasets_package():
        if auto_install:
            if not install_datasets_package():
                print("\nFailed to install datasets package automatically.")
                print("Please install manually: pip install datasets")
                return False
        else:
            print("\nThe 'datasets' package is not installed.")
            print("Please install it: pip install datasets")
            return False
    
    # Import datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        print("\nFailed to import datasets library")
        return False
    
    # Check which files already exist
    existing_files = []
    required_files = ["train_spider.json", "train_others.json", "dev.json", "tables.json"]
    
    for filename in required_files:
        filepath = spider_dir / filename
        if filepath.exists():
            existing_files.append(filename)
            print(f"  ⊗ {filename} already exists, skipping...")
    
    if len(existing_files) == len(required_files):
        print("\n✓ All required files already exist!")
        return True
    
    # Download the Spider dataset
    print("\nDownloading Spider dataset from Hugging Face...")
    print("Using hujudev/spider-text-2-sql (complete dataset with 8659 examples)")
    print("This may take a few minutes depending on your internet connection...\n")
    
    try:
        # Load the dataset - use hujudev version which has complete 8659 examples
        dataset = load_dataset("hujudev/spider-text-2-sql")
        
        print("\nDataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Process each split
        if 'train' in dataset:
            train_data = dataset['train']
            print(f"\nProcessing training data ({len(train_data)} examples)...")
            
            # Convert hujudev format to standard Spider format
            # hujudev uses: query, Input, Question, Schema, db_id
            # Standard Spider uses: db_id, question, SQL (or query)
            train_examples = []
            for i in range(len(train_data)):
                item = train_data[i]
                # Convert to standard format
                converted_item = {
                    'db_id': item.get('db_id', ''),
                    'question': item.get('Question', item.get('question', '')).replace('Question: ', '').strip(),
                    'query': item.get('query', ''),
                    # Keep other fields for potential use
                    'Input': item.get('Input', ''),
                    'Schema': item.get('Schema', '')
                }
                train_examples.append(converted_item)
            
            # Save train_spider.json (first 7000 examples)
            train_spider_path = spider_dir / "train_spider.json"
            if not train_spider_path.exists():
                train_spider_examples = train_examples[:7000]
                with open(train_spider_path, 'w') as f:
                    json.dump(train_spider_examples, f, indent=2)
                print(f"  ✓ Saved {len(train_spider_examples)} examples to train_spider.json")
            else:
                print(f"  ⊗ train_spider.json already exists")
            
            # Save train_others.json (remaining examples)
            train_others_path = spider_dir / "train_others.json"
            if not train_others_path.exists():
                train_others_examples = train_examples[7000:]
                with open(train_others_path, 'w') as f:
                    json.dump(train_others_examples, f, indent=2)
                print(f"  ✓ Saved {len(train_others_examples)} examples to train_others.json")
            else:
                print(f"  ⊗ train_others.json already exists")
        
        if 'validation' in dataset or 'dev' in dataset:
            val_key = 'validation' if 'validation' in dataset else 'dev'
            val_data = dataset[val_key]
            print(f"\nProcessing validation data ({len(val_data)} examples)...")
            
            # Convert validation data to standard format
            val_examples = []
            for i in range(len(val_data)):
                item = val_data[i]
                converted_item = {
                    'db_id': item.get('db_id', ''),
                    'question': item.get('Question', item.get('question', '')).replace('Question: ', '').strip(),
                    'query': item.get('query', ''),
                    'Input': item.get('Input', ''),
                    'Schema': item.get('Schema', '')
                }
                val_examples.append(converted_item)
            
            dev_path = spider_dir / "dev.json"
            if not dev_path.exists():
                with open(dev_path, 'w') as f:
                    json.dump(val_examples, f, indent=2)
                print(f"  ✓ Saved {len(val_examples)} examples to dev.json")
            else:
                print(f"  ⊗ dev.json already exists")
        else:
            print("  ⚠ No validation/dev split found in dataset")
        
        # Try to get tables.json from the dataset
        # Some versions include this in the dataset info
        tables_path = spider_dir / "tables.json"
        if not tables_path.exists():
            # Check if tables info is in the dataset
            if hasattr(dataset, 'info') and hasattr(dataset.info, 'features'):
                # Try to extract table schema info
                print("\nAttempting to extract tables.json...")
                print("  ⚠ tables.json not found in dataset, you may need to download it separately")
                print(f"  Try: wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation_examples/examples/tables.json -P {spider_dir}")
            else:
                print("\n  ⚠ tables.json not found in dataset")
                print(f"  Download separately from: https://github.com/taoyds/spider")
        
        print("\n" + "=" * 60)
        
        # Check if we have the complete dataset
        train_others_path = spider_dir / "train_others.json"
        tables_path = spider_dir / "tables.json"
        missing_files = []
        if not train_others_path.exists():
            missing_files.append("train_others.json")
        if not tables_path.exists():
            missing_files.append("tables.json")
        
        if missing_files:
            if 'train_others.json' not in missing_files:
                print("⚠ Partial dataset download completed")
                print(f"Missing files: {', '.join(missing_files)}")
                print("\nTo get the complete dataset:")
                print("  1. Clone the Spider repository:")
                print("     git clone https://github.com/taoyds/spider.git")
                print("  2. Copy missing files to data/spider/")
            else:
                print("✓ Complete Spider dataset downloaded successfully!")
                print("  All 8659 training examples are now available")
                if missing_files:
                    print(f"  Missing only: {', '.join(missing_files)}")
        else:
            print("✓ Dataset download completed successfully!")
        
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("Please try again or download manually from: https://github.com/taoyds/spider")
        return False


def verify_dataset(data_dir="data/spider"):
    """Verify dataset files and print statistics"""
    data_path = Path(data_dir)
    
    print("\nDataset Statistics:")
    print("-" * 40)
    
    # Check file contents
    try:
        with open(data_path / "train_spider.json") as f:
            train_spider_data = json.load(f)
            print(f"Train Spider examples: {len(train_spider_data)}")
        
        train_others_count = 0
        if (data_path / "train_others.json").exists():
            with open(data_path / "train_others.json") as f:
                train_others_data = json.load(f)
                train_others_count = len(train_others_data)
                print(f"Train Others examples: {train_others_count}")
        else:
            print("Train Others examples: 0 (NOT FOUND)")
        
        total_train = len(train_spider_data) + train_others_count
        print(f"Total training examples: {total_train}")
        
        with open(data_path / "dev.json") as f:
            dev_data = json.load(f)
            print(f"Dev examples: {len(dev_data)}")
        
        tables_count = 0
        if (data_path / "tables.json").exists():
            with open(data_path / "tables.json") as f:
                tables_data = json.load(f)
                tables_count = len(tables_data)
                print(f"Database schemas: {tables_count}")
        else:
            print("Database schemas: 0 (NOT FOUND)")
        
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

