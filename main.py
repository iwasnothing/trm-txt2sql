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
    train_parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    train_parser.add_argument('--label_smoothing', type=float, default=0.0, 
                             help='Label smoothing factor (0.0 = no smoothing, 0.1 = typical smoothing)')
    train_parser.add_argument('--early_stopping_patience', type=int, default=3,
                             help='Number of epochs to wait before early stopping')
    train_parser.add_argument('--use_constrained_decoding', action='store_true',
                             help='Enable SQL grammar-constrained decoding during generation')
    train_parser.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu',
                             help='Device to use')
    train_parser.add_argument('--log_dir', default='runs', help='Directory for TensorBoard logs')
    train_parser.add_argument('--experiment_name', default=None,
                             help='Experiment name for TensorBoard (default: timestamp)')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    eval_parser.add_argument('--data_dir', default='data/spider', help='Data directory')
    eval_parser.add_argument('--processed_dir', default='data/processed', help='Processed data directory')
    eval_parser.add_argument('--generated_db_dir', default='data/generated_databases',
                             help='Directory containing generated SQLite databases')
    eval_parser.add_argument('--output_file', default='evaluation_results.json', help='Output file')
    eval_parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    eval_parser.add_argument('--max_len', type=int, default=512, help='Maximum sequence length')
    eval_parser.add_argument('--use_constrained_decoding', action='store_true',
                             help='Enable SQL grammar-constrained decoding during generation')
    eval_parser.add_argument('--device', default='cuda' if __import__('torch').cuda.is_available() else 'cpu',
                             help='Device to use')
    
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
        # Build argument list, handling boolean flags correctly
        arg_list = ['train']
        for k, v in vars(args).items():
            if k == 'command':
                continue
            if v is None:
                continue
            # Handle boolean flags (store_true): only include if True
            if isinstance(v, bool):
                if v:  # Only add flag if True
                    arg_list.append(f'--{k}')
            else:
                arg_list.append(f'--{k}={v}')
        sys.argv = arg_list
        train_main()
    
    elif args.command == 'evaluate':
        from evaluate import main as eval_main
        # Build argument list, handling boolean flags correctly
        arg_list = ['evaluate']
        for k, v in vars(args).items():
            if k == 'command':
                continue
            if v is None:
                continue
            # Handle boolean flags (store_true): only include if True
            if isinstance(v, bool):
                if v:  # Only add flag if True
                    arg_list.append(f'--{k}')
            else:
                arg_list.append(f'--{k}={v}')
        sys.argv = arg_list
        eval_main()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

