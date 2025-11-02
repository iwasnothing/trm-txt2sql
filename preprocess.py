#!/usr/bin/env python3
"""
Preprocess Spider dataset for TRM model training
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class SpiderPreprocessor:
    """Preprocess Spider dataset for text-to-SQL training"""
    
    def __init__(self, data_dir="data/spider"):
        self.data_dir = Path(data_dir)
        self.tables = self._load_tables()
        
    def _load_tables(self) -> Dict:
        """Load database schema information"""
        tables_path = self.data_dir / "tables.json"
        if not tables_path.exists():
            return {}
        
        with open(tables_path, 'r') as f:
            tables_data = json.load(f)
        
        # Create a mapping from db_id to schema
        schema_dict = {}
        for db_info in tables_data:
            db_id = db_info['db_id']
            schema_dict[db_id] = {
                'tables': db_info.get('table_names_original', []),
                'columns': db_info.get('column_names_original', []),
                'column_types': db_info.get('column_types', []),
                'primary_keys': db_info.get('primary_keys', []),
                'foreign_keys': db_info.get('foreign_keys', [])
            }
        
        return schema_dict
    
    def normalize_sql(self, sql: str) -> str:
        """Normalize SQL query"""
        # Remove extra whitespace
        sql = re.sub(r'\s+', ' ', sql)
        # Uppercase SQL keywords
        sql = re.sub(r'\b(SELECT|FROM|WHERE|JOIN|ON|GROUP|BY|ORDER|HAVING|AS|AND|OR|NOT|IN|EXISTS|COUNT|SUM|AVG|MAX|MIN)\b', 
                    lambda m: m.group(1).upper(), sql, flags=re.IGNORECASE)
        return sql.strip()
    
    def format_schema(self, db_id: str) -> str:
        """Format database schema as text"""
        if db_id not in self.tables:
            return f"Database: {db_id}"
        
        schema = self.tables[db_id]
        schema_text = f"Database: {db_id}\n"
        
        # Group columns by table
        table_columns = defaultdict(list)
        for i, (table_idx, col_name) in enumerate(schema['columns']):
            if table_idx == -1:
                continue
            table_name = schema['tables'][table_idx]
            col_type = schema['column_types'][i] if i < len(schema['column_types']) else 'text'
            table_columns[table_name].append(f"{col_name} ({col_type})")
        
        # Format tables
        for table_name in schema['tables']:
            schema_text += f"Table: {table_name}\n"
            if table_name in table_columns:
                for col_info in table_columns[table_name]:
                    schema_text += f"  - {col_info}\n"
        
        return schema_text
    
    def create_examples(self, data_file: str) -> List[Dict]:
        """Create training examples from Spider data file"""
        file_path = self.data_dir / data_file
        if not file_path.exists():
            print(f"File not found: {file_path}")
            return []
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        examples = []
        for item in data:
            db_id = item.get('db_id', '')
            question = item.get('question', '')
            sql = item.get('SQL', item.get('query', ''))
            
            # Format schema context
            schema_context = self.format_schema(db_id)
            
            # Normalize SQL
            normalized_sql = self.normalize_sql(sql)
            
            example = {
                'question': question,
                'sql': normalized_sql,
                'db_id': db_id,
                'schema': schema_context,
                'original_sql': sql
            }
            
            examples.append(example)
        
        return examples
    
    def prepare_train_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Prepare training and validation data"""
        # Load training data
        train_examples = []
        
        # Try train_spider.json first
        train_spider = self.create_examples("train_spider.json")
        train_examples.extend(train_spider)
        
        # Try train_others.json if available
        train_others = self.create_examples("train_others.json")
        train_examples.extend(train_others)
        
        # Load validation data
        val_examples = self.create_examples("dev.json")
        
        print(f"Loaded {len(train_examples)} training examples")
        print(f"Loaded {len(val_examples)} validation examples")
        
        return train_examples, val_examples
    
    def save_preprocessed(self, train_data: List[Dict], val_data: List[Dict], 
                         output_dir="data/processed"):
        """Save preprocessed data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Save training data
        train_path = output_path / "train.json"
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        print(f"Saved {len(train_data)} training examples to {train_path}")
        
        # Save validation data
        val_path = output_path / "val.json"
        with open(val_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        print(f"Saved {len(val_data)} validation examples to {val_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess Spider dataset")
    parser.add_argument("--data_dir", default="data/spider", help="Input data directory")
    parser.add_argument("--output_dir", default="data/processed", help="Output directory")
    args = parser.parse_args()
    
    preprocessor = SpiderPreprocessor(args.data_dir)
    train_data, val_data = preprocessor.prepare_train_data()
    preprocessor.save_preprocessed(train_data, val_data, args.output_dir)

