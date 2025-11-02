#!/usr/bin/env python3
"""
Main entry point for TRM Text-to-SQL training and evaluation
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="TRM Text-to-SQL: Complete Pipeline")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download Spider dataset')
    download_parser.add_argument('--data_dir', default='data', help='Data directory')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess dataset')
    preprocess_parser.add_argument('--data_dir', default='data/spider', help='Input data directory')
    preprocess_parser.add_argument('--output_dir', default='data/processed', help='Output directory')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--data_dir', default='data/spider', help='Data directory')
    train_parser.add_argument('--processed_dir', default='data/processed', help='Processed data directory')
    train_parser.add_argument('--output_dir', default='checkpoints', help='Output directory')
    train_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    train_parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    train_parser.add_argument('--n_layers', type=int, default=6, help='Number of layers')
    train_parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--data_dir', default='data/spider', help='Data directory')
    eval_parser.add_argument('--processed_dir', default='data/processed', help='Processed data directory')
    eval_parser.add_argument('--output_file', default='evaluation_results.json', help='Output file')
    eval_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    eval_parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    if args.command == 'download':
        from download_spider import download_spider_dataset, verify_dataset
        if download_spider_dataset(args.data_dir):
            verify_dataset(f"{args.data_dir}/spider")
    
    elif args.command == 'preprocess':
        from preprocess import SpiderPreprocessor
        preprocessor = SpiderPreprocessor(args.data_dir)
        train_data, val_data = preprocessor.prepare_train_data()
        preprocessor.save_preprocessed(train_data, val_data, args.output_dir)
    
    elif args.command == 'train':
        from train import main as train_main
        sys.argv = ['train'] + [f'--{k}={v}' for k, v in vars(args).items() 
                                if k != 'command' and v is not None]
        train_main()
    
    elif args.command == 'evaluate':
        from evaluate import main as eval_main
        sys.argv = ['evaluate'] + [f'--{k}={v}' for k, v in vars(args).items() 
                                  if k != 'command' and v is not None]
        eval_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

