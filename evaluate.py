#!/usr/bin/env python3
"""
Evaluation script for TRM Text-to-SQL model
"""
import torch
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import re
import sqlite3

from trm_model import TRMTextToSQL
from tokenizer import SimpleTokenizer
from train import TextToSQLDataset
from torch.utils.data import DataLoader

# Import clean_identifier from generate_random_data
import sys
sys.path.insert(0, str(Path(__file__).parent))
from generate_random_data import clean_identifier


def compute_exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Compute exact match accuracy"""
    # Normalize SQL queries
    pred_sql = normalize_sql(pred_sql)
    gold_sql = normalize_sql(gold_sql)
    return pred_sql.strip().lower() == gold_sql.strip().lower()


def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison"""
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql)
    # Uppercase SQL keywords
    sql = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|ON|GROUP|BY|ORDER|HAVING|AS|AND|OR|NOT|IN|EXISTS|COUNT|SUM|AVG|MAX|MIN)\b',
                lambda m: m.group(1).upper(), sql, flags=re.IGNORECASE)
    # Remove trailing semicolons
    sql = sql.rstrip(';')
    return sql.strip()


def get_database_schema(conn: sqlite3.Connection) -> dict:
    """Get table and column names from database"""
    schema = {'tables': {}, 'all_columns': set()}
    
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for row in cursor.fetchall():
        table_name = row[0]
        schema['tables'][table_name.lower()] = table_name  # Store both lower and original
        
        # Get columns for this table
        cursor2 = conn.execute(f'PRAGMA table_info("{table_name}")')
        for col_row in cursor2.fetchall():
            col_name = col_row[1]
            schema['all_columns'].add(col_name.lower())
            schema['all_columns'].add(col_name)  # Also store original case
    
    return schema


def normalize_sql_for_database(sql: str, schema: dict) -> str:
    """
    Normalize SQL query to work with generated database.
    Maps table names to actual database table names (handles cleaned identifiers).
    """
    # Remove semicolons
    sql = sql.rstrip(';').strip()
    
    # Map table names from SQL to actual database table names
    table_map = schema.get('tables', {})
    
    # Try to replace table names (case-insensitive match)
    def replace_table_name(match):
        table_ref = match.group(1).strip('"\'')  # Remove quotes if present
        table_lower = table_ref.lower()
        
        # First try exact match (case-insensitive)
        if table_lower in table_map:
            quoted_name = f'"{table_map[table_lower]}"'
            return match.group(0).replace(table_ref, quoted_name, 1)
        
        # Try applying cleaning logic to find match
        cleaned_table = clean_identifier(table_ref, prefix="tbl_")
        if cleaned_table.lower() in table_map:
            quoted_name = f'"{table_map[cleaned_table.lower()]}"'
            return match.group(0).replace(table_ref, quoted_name, 1)
        
        # If no match found, quote the original name
        return match.group(0).replace(table_ref, f'"{table_ref}"', 1)
    
    # Replace FROM clauses (handle quoted and unquoted names)
    sql = re.sub(r'\bFROM\s+(["\']?\w+["\']?)', replace_table_name, sql, flags=re.IGNORECASE)
    # Replace JOIN clauses
    sql = re.sub(r'\bJOIN\s+(["\']?\w+["\']?)', replace_table_name, sql, flags=re.IGNORECASE)
    # Replace UPDATE clauses
    sql = re.sub(r'\bUPDATE\s+(["\']?\w+["\']?)', replace_table_name, sql, flags=re.IGNORECASE)
    # Replace INSERT INTO clauses
    sql = re.sub(r'\bINSERT\s+INTO\s+(["\']?\w+["\']?)', replace_table_name, sql, flags=re.IGNORECASE)
    
    return sql


def compute_execution_accuracy(pred_sql: str, gold_sql: str, db_id: str, 
                               generated_db_dir: str = "data/generated_databases") -> bool:
    """
    Compute execution accuracy by executing SQL queries on generated databases
    
    Args:
        pred_sql: Predicted SQL query
        gold_sql: Gold/ground truth SQL query
        db_id: Database identifier
        generated_db_dir: Directory containing generated SQLite databases
    
    Returns:
        True if both queries return the same results, False otherwise
    """
    db_path = Path(generated_db_dir) / f"{db_id}.db"
    
    if not db_path.exists():
        # Database doesn't exist, cannot compute execution accuracy
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries for easier comparison
        
        # Get actual schema from database
        schema = get_database_schema(conn)
        
        # Normalize SQL queries to use actual database table names
        pred_sql_normalized = normalize_sql_for_database(pred_sql, schema)
        gold_sql_normalized = normalize_sql_for_database(gold_sql, schema)
        
        # Execute predicted SQL
        try:
            pred_cursor = conn.execute(pred_sql_normalized)
            pred_results = [dict(row) for row in pred_cursor.fetchall()]
            pred_cols = [desc[0] for desc in pred_cursor.description] if pred_cursor.description else []
        except sqlite3.Error as e:
            # Predicted SQL failed to execute
            conn.close()
            return False
        
        # Execute gold SQL
        try:
            gold_cursor = conn.execute(gold_sql_normalized)
            gold_results = [dict(row) for row in gold_cursor.fetchall()]
            gold_cols = [desc[0] for desc in gold_cursor.description] if gold_cursor.description else []
        except sqlite3.Error as e:
            # Gold SQL failed to execute
            conn.close()
            return False
        
        conn.close()
        
        # Compare results
        # Normalize column names (case-insensitive) and compare
        def normalize_row(row_dict, cols):
            # Sort by column names and convert values to comparable types
            normalized = {}
            for col in cols:
                val = row_dict.get(col)
                # Convert to string for comparison (handles None, numbers, etc.)
                if val is None:
                    normalized[col.lower()] = None
                else:
                    normalized[col.lower()] = str(val).lower() if isinstance(val, str) else val
            return tuple(sorted(normalized.items()))
        
        # Normalize and compare result sets
        pred_normalized = sorted([normalize_row(row, pred_cols) for row in pred_results])
        gold_normalized = sorted([normalize_row(row, gold_cols) for row in gold_results])
        
        return pred_normalized == gold_normalized
        
    except Exception as e:
        # Any other error - return False
        return False


def evaluate_model(model, dataloader, question_tokenizer, sql_tokenizer, device, max_len=512,
                   generated_db_dir: str = "data/generated_databases"):
    """Evaluate model and compute metrics"""
    model.eval()
    
    exact_matches = 0
    execution_acc = 0
    total = 0
    
    results = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            question = batch['question'].to(device)
            question_mask = (question != 0).float()
            
            # Generate SQL
            generated_sql = model.generate(
                question, 
                question_mask,
                max_len=max_len,
                start_token=1,
                end_token=2
            )
            
            # Decode generated SQL
            for i in range(generated_sql.size(0)):
                sql_ids = generated_sql[i].cpu().tolist()
                # Remove padding and end token
                sql_ids = [tid for tid in sql_ids if tid != 0 and tid != 2]
                pred_sql = sql_tokenizer.decode(sql_ids[1:])  # Skip start token
                
                gold_sql = batch['sql_text'][i]
                db_id = batch['db_id'][i]
                
                # Compute metrics
                em = compute_exact_match(pred_sql, gold_sql)
                exec_acc = compute_execution_accuracy(pred_sql, gold_sql, db_id, generated_db_dir)
                
                exact_matches += em
                execution_acc += exec_acc
                total += 1
                
                results.append({
                    'question': batch['question_text'][i],
                    'gold_sql': gold_sql,
                    'pred_sql': pred_sql,
                    'exact_match': em,
                    'execution_accuracy': exec_acc,
                    'db_id': db_id
                })
    
    exact_match_accuracy = exact_matches / total if total > 0 else 0
    execution_accuracy = execution_acc / total if total > 0 else 0
    
    return exact_match_accuracy, execution_accuracy, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM Text-to-SQL model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", default="data/spider", help="Data directory")
    parser.add_argument("--processed_dir", default="data/processed", help="Processed data directory")
    parser.add_argument("--generated_db_dir", default="data/generated_databases", 
                       help="Directory containing generated SQLite databases")
    parser.add_argument("--output_file", default="evaluation_results.json", help="Output file for results")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    
    # Reconstruct tokenizers
    question_tokenizer = SimpleTokenizer()
    question_tokenizer.vocab = dict(checkpoint['question_tokenizer_vocab'])
    question_tokenizer._build_vocab = False
    question_tokenizer.inv_vocab = {v: k for k, v in question_tokenizer.vocab.items()}
    
    sql_tokenizer = SimpleTokenizer()
    sql_tokenizer.vocab = dict(checkpoint['sql_tokenizer_vocab'])
    sql_tokenizer._build_vocab = False
    sql_tokenizer.inv_vocab = {v: k for k, v in sql_tokenizer.vocab.items()}
    
    question_vocab_size = len(question_tokenizer.vocab)
    sql_vocab_size = len(sql_tokenizer.vocab)
    
    # Load model
    device = torch.device(args.device)
    model = TRMTextToSQL(
        question_vocab_size=question_vocab_size,
        sql_vocab_size=sql_vocab_size,
        d_model=checkpoint.get('d_model', 512),
        n_heads=checkpoint.get('n_heads', 8),
        n_encoder_layers=checkpoint.get('n_layers', 6),
        n_decoder_layers=checkpoint.get('n_layers', 6),
        max_len=args.max_len
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load validation data
    processed_path = Path(args.processed_dir)
    with open(processed_path / "val.json") as f:
        val_examples = json.load(f)
    
    print(f"Loaded {len(val_examples)} validation examples")
    
    # Create dataset and dataloader
    val_dataset = TextToSQLDataset(val_examples, question_tokenizer, sql_tokenizer, args.max_len)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate
    print("\nEvaluating model...")
    print(f"Using generated databases from: {args.generated_db_dir}")
    exact_match_acc, exec_acc, results = evaluate_model(
        model, val_loader, question_tokenizer, sql_tokenizer, device, args.max_len,
        generated_db_dir=args.generated_db_dir
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"Exact Match Accuracy: {exact_match_acc*100:.2f}%")
    print(f"Execution Accuracy: {exec_acc*100:.2f}%")
    print(f"Total examples: {len(results)}")
    print("="*50)
    
    # Save detailed results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            'exact_match_accuracy': exact_match_acc,
            'execution_accuracy': exec_acc,
            'total_examples': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")
    
    # Show some examples
    print("\nSample predictions:")
    print("-"*50)
    for i, result in enumerate(results[:5]):
        print(f"\nExample {i+1}:")
        print(f"Question: {result['question']}")
        print(f"Gold SQL: {result['gold_sql']}")
        print(f"Pred SQL: {result['pred_sql']}")
        print(f"Exact Match: {result['exact_match']}")


if __name__ == "__main__":
    main()

