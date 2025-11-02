#!/usr/bin/env python3
"""
Training script for TRM Text-to-SQL model
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from datetime import datetime

from trm_model import TRMTextToSQL
from preprocess import SpiderPreprocessor
from tokenizer import SimpleTokenizer


class TextToSQLDataset(Dataset):
    """Dataset for Text-to-SQL training"""
    
    def __init__(self, examples: list, question_tokenizer, sql_tokenizer, max_len=512):
        self.examples = examples
        self.question_tokenizer = question_tokenizer
        self.sql_tokenizer = sql_tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize question (with schema context)
        question_text = f"{example['schema']}\nQuestion: {example['question']}"
        question_tokens = self.question_tokenizer.encode(question_text, max_length=self.max_len, 
                                                          truncation=True, padding='max_length')
        
        # Tokenize SQL
        sql_tokens = self.sql_tokenizer.encode(example['sql'], max_length=self.max_len,
                                               truncation=True, padding='max_length')
        
        return {
            'question': torch.tensor(question_tokens, dtype=torch.long),
            'sql': torch.tensor(sql_tokens, dtype=torch.long),
            'question_text': example['question'],
            'sql_text': example['sql'],
            'db_id': example['db_id']
        }


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None, global_step=0):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        question = batch['question'].to(device)
        sql = batch['sql'].to(device)
        
        # Create masks (0 for padding)
        question_mask = (question != 0).float()
        sql_mask = (sql != 0).float()
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(question, sql, question_mask, sql_mask)
        # logits: [batch_size, seq_len_sql, vocab_size]
        
        # For teacher forcing: predict next token at each position
        # Input: [<start>, token1, token2, ..., tokenN]
        # Logits predict: [token1, token2, ..., tokenN, <next>] (but we only need first N)
        # Targets: [token1, token2, ..., tokenN]
        # So we shift both: logits[:, :-1] and targets = sql[:, 1:]
        logits = logits[:, :-1, :].contiguous()  # Remove last position [batch_size, seq_len-1, vocab_size]
        targets = sql[:, 1:].contiguous()  # Shift by 1 for next-token prediction [batch_size, seq_len-1]
        
        # Reshape for loss calculation
        logits = logits.view(-1, logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
        targets = targets.view(-1)  # [batch_size * (seq_len-1)]
        
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Log to TensorBoard
        current_step = global_step + batch_idx
        if writer is not None:
            writer.add_scalar('Loss/Train_Batch', loss.item(), current_step)
            # Log learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('Learning_Rate', current_lr, current_step)
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches
    if writer is not None:
        writer.add_scalar('Loss/Train_Epoch', avg_loss, epoch)
    
    return avg_loss


def evaluate(model, dataloader, criterion, device, epoch=None, writer=None):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question = batch['question'].to(device)
            sql = batch['sql'].to(device)
            
            question_mask = (question != 0).float()
            sql_mask = (sql != 0).float()
            
            logits = model(question, sql, question_mask, sql_mask)
            # Align logits and targets for next-token prediction
            logits = logits[:, :-1, :].contiguous()  # [batch_size, seq_len-1, vocab_size]
            targets = sql[:, 1:].contiguous()  # [batch_size, seq_len-1]
            
            # Reshape for loss calculation
            logits = logits.view(-1, logits.size(-1))  # [batch_size * (seq_len-1), vocab_size]
            targets = targets.view(-1)  # [batch_size * (seq_len-1)]
            
            loss = criterion(logits, targets)
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Log to TensorBoard
    if writer is not None and epoch is not None:
        writer.add_scalar('Loss/Validation_Epoch', avg_loss, epoch)
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train TRM Text-to-SQL model")
    parser.add_argument("--data_dir", default="data/spider", help="Data directory")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--output_dir", default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of layers")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--label_smoothing", type=float, default=0.0, 
                       help="Label smoothing factor (0.0 = no smoothing, 0.1 = typical smoothing)")
    parser.add_argument("--early_stopping_patience", type=int, default=3, 
                       help="Number of epochs to wait before early stopping")
    parser.add_argument("--use_constrained_decoding", action="store_true",
                       help="Enable SQL grammar-constrained decoding during generation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    parser.add_argument("--log_dir", default="runs", help="Directory for TensorBoard logs")
    parser.add_argument("--experiment_name", default=None, help="Experiment name for TensorBoard (default: timestamp)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Setup TensorBoard logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    # Create experiment name with timestamp if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"trm_text2sql_{timestamp}"
    else:
        experiment_name = args.experiment_name
    
    log_path = log_dir / experiment_name
    writer = SummaryWriter(log_dir=str(log_path))
    print(f"TensorBoard logs will be saved to: {log_path}")
    print(f"To view: tensorboard --logdir {log_dir}")
    
    # Load preprocessed data
    processed_path = Path(args.processed_dir)
    if not (processed_path / "train.json").exists():
        print("Preprocessed data not found. Running preprocessing...")
        preprocessor = SpiderPreprocessor(args.data_dir)
        train_examples, val_examples = preprocessor.prepare_train_data()
        preprocessor.save_preprocessed(train_examples, val_examples, args.processed_dir)
    else:
        with open(processed_path / "train.json") as f:
            train_examples = json.load(f)
        with open(processed_path / "val.json") as f:
            val_examples = json.load(f)
    
    print(f"Loaded {len(train_examples)} training examples")
    print(f"Loaded {len(val_examples)} validation examples")
    
    # Initialize tokenizers
    question_tokenizer = SimpleTokenizer()
    sql_tokenizer = SimpleTokenizer()
    
    # Build vocabulary from training data
    print("Building vocabulary...")
    for example in train_examples:
        question_text = f"{example['schema']}\nQuestion: {example['question']}"
        question_tokenizer.encode(question_text, max_length=args.max_len)
        sql_tokenizer.encode(example['sql'], max_length=args.max_len)
    
    question_tokenizer._build_vocab = False
    sql_tokenizer._build_vocab = False
    
    question_vocab_size = question_tokenizer.get_vocab_size()
    sql_vocab_size = sql_tokenizer.get_vocab_size()
    
    print(f"Question vocab size: {question_vocab_size}")
    print(f"SQL vocab size: {sql_vocab_size}")
    
    # Create datasets
    train_dataset = TextToSQLDataset(train_examples, question_tokenizer, sql_tokenizer, args.max_len)
    val_dataset = TextToSQLDataset(val_examples, question_tokenizer, sql_tokenizer, args.max_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    device = torch.device(args.device)
    model = TRMTextToSQL(
        question_vocab_size=question_vocab_size,
        sql_vocab_size=sql_vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_layers,
        n_decoder_layers=args.n_layers,
        max_len=args.max_len,
        dropout=args.dropout,
        use_constrained_decoding=args.use_constrained_decoding,
        sql_tokenizer=sql_tokenizer if args.use_constrained_decoding else None
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Training loop
    best_val_loss = float('inf')
    patience = args.early_stopping_patience  # Number of epochs to wait before stopping
    patience_counter = 0
    global_step = 0
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, 
                                writer=writer, global_step=global_step)
        val_loss = evaluate(model, val_loader, criterion, device, epoch=epoch, writer=writer)
        
        # Update global step counter
        global_step += len(train_loader)
        
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'd_model': args.d_model,
            'n_heads': args.n_heads,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'question_tokenizer_vocab': dict(question_tokenizer.vocab),
            'sql_tokenizer_vocab': dict(sql_tokenizer.vocab),
        }
        
        checkpoint_path = output_path / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_path / "best_model.pt"
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
            patience_counter = 0  # Reset counter
            if writer is not None:
                writer.add_scalar('Best_Validation_Loss', best_val_loss, epoch)
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    # Log hyperparameters at the end with final metrics
    if writer is not None:
        writer.add_hparams(
            {
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'd_model': args.d_model,
                'n_heads': args.n_heads,
                'n_layers': args.n_layers,
                'max_len': args.max_len,
                'dropout': args.dropout,
                'label_smoothing': args.label_smoothing,
                'epochs': args.epochs,
                'early_stopping_patience': args.early_stopping_patience,
            },
            {
                'final_train_loss': train_loss,
                'final_val_loss': val_loss,
                'best_val_loss': best_val_loss,
            }
        )
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
    
    print("\nTraining complete!")
    print(f"TensorBoard logs saved to: {log_path}")


if __name__ == "__main__":
    main()

