#!/usr/bin/env python3
"""
Generate random data for SQLite databases based on schemas from val.json
Creates tables and inserts 1000 rows using recursive CTEs
"""
import json
import sqlite3
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple, Set
from collections import defaultdict, OrderedDict


class SchemaParser:
    """Parse schema text into structured format"""
    
    def __init__(self):
        self.tables = OrderedDict()  # table_name -> columns
        self.db_name = None
        
    def parse(self, schema_text: str) -> Dict:
        """Parse schema text into tables and columns"""
        self.tables = OrderedDict()
        lines = schema_text.strip().split('\n')
        
        # Extract database name
        if lines[0].startswith('Database:'):
            self.db_name = lines[0].split('Database:')[1].strip()
        
        current_table = None
        for line in lines[1:]:
            line = line.strip()
            if line.startswith('Table:'):
                current_table = line.split('Table:')[1].strip()
                self.tables[current_table] = []
            elif line.startswith('-') and current_table:
                # Extract column name and type
                col_line = line[1:].strip()
                match = re.match(r'(.+?)\s*\((.+?)\)', col_line)
                if match:
                    col_name = match.group(1).strip()
                    col_type = match.group(2).strip().lower()
                    self.tables[current_table].append({
                        'name': col_name,
                        'type': col_type,
                        'is_id': self._is_id_column(col_name)
                    })
        
        return {
            'db_name': self.db_name,
            'tables': self.tables
        }
    
    def _is_id_column(self, col_name: str) -> bool:
        """Check if column is likely an ID/Primary Key"""
        col_lower = col_name.lower()
        return ('_id' in col_lower or col_lower.endswith('id')) and \
               not col_lower.startswith('foreign')


def clean_identifier(name: str, prefix: str = "") -> str:
    """Clean identifier name for SQLite - handle numbers, special chars, reserved names"""
    # Replace special chars with underscore
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    
    # If starts with digit, add prefix
    if cleaned and cleaned[0].isdigit():
        cleaned = f"{prefix}{cleaned}" if prefix else f"col_{cleaned}"
    
    # SQLite reserves names starting with "sqlite_" - avoid conflicts
    if cleaned.lower().startswith('sqlite_'):
        cleaned = f"{prefix}reserved_{cleaned}" if prefix else f"tbl_reserved_{cleaned}"
    
    # Also check for other reserved words (basic check)
    reserved_words = {'rowid', 'oid', '_rowid_', 'sqlite_sequence', 'sqlite_stat1', 
                     'sqlite_stat2', 'sqlite_stat3', 'sqlite_stat4'}
    if cleaned.lower() in reserved_words:
        cleaned = f"{prefix}reserved_{cleaned}" if prefix else f"tbl_reserved_{cleaned}"
    
    return cleaned


class DataGenerator:
    """Generate random data for SQLite tables"""
    
    def __init__(self, schema_info: Dict, tables_json_path: str = None):
        self.schema_info = schema_info
        self.db_name = schema_info['db_name']
        self.tables = schema_info['tables']
        self.tables_json = None
        self.foreign_keys = defaultdict(list)
        self.primary_keys = defaultdict(list)
        
        # Load foreign key information from tables.json if available
        if tables_json_path and Path(tables_json_path).exists():
            self._load_foreign_keys(tables_json_path)
        
        # Sample text values
        self.text_samples = {
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry'],
            'country': ['USA', 'UK', 'Canada', 'France', 'Germany', 'Japan', 'China', 'India'],
            'city': ['New York', 'London', 'Paris', 'Tokyo', 'Berlin', 'Toronto', 'Sydney', 'Mumbai'],
            'location': ['North', 'South', 'East', 'West', 'Central', 'Upper', 'Lower'],
            'theme': ['Rock', 'Pop', 'Jazz', 'Classical', 'Electronic', 'Folk', 'Blues', 'Country'],
            'song': ['Song A', 'Song B', 'Song C', 'Song D', 'Song E', 'Song F'],
        }
    
    def _load_foreign_keys(self, tables_json_path: str):
        """Load foreign key relationships from tables.json"""
        try:
            with open(tables_json_path, 'r') as f:
                tables_data = json.load(f)
            
            # Find matching database
            for db_info in tables_data:
                if db_info.get('db_id') == self.db_name:
                    # Map foreign keys
                    if 'foreign_keys' in db_info:
                        column_names = db_info.get('column_names_original', db_info.get('column_names', []))
                        table_names = db_info.get('table_names_original', db_info.get('table_names', []))
                        
                        for fk_pair in db_info['foreign_keys']:
                            child_idx, parent_idx = fk_pair
                            if child_idx < len(column_names) and parent_idx < len(column_names):
                                child_col = column_names[child_idx][1]
                                parent_col = column_names[parent_idx][1]
                                child_table_idx = column_names[child_idx][0]
                                parent_table_idx = column_names[parent_idx][0]
                                
                                if child_table_idx >= 0 and parent_table_idx >= 0:
                                    child_table = table_names[child_table_idx]
                                    parent_table = table_names[parent_table_idx]
                                    self.foreign_keys[child_table].append({
                                        'child_col': child_col,
                                        'parent_table': parent_table,
                                        'parent_col': parent_col
                                    })
                    
                    # Map primary keys
                    if 'primary_keys' in db_info:
                        column_names = db_info.get('column_names_original', db_info.get('column_names', []))
                        table_names = db_info.get('table_names_original', db_info.get('table_names', []))
                        
                        for pk_idx in db_info['primary_keys']:
                            if pk_idx < len(column_names):
                                table_idx = column_names[pk_idx][0]
                                col_name = column_names[pk_idx][1]
                                if table_idx >= 0:
                                    table_name = table_names[table_idx]
                                    self.primary_keys[table_name].append(col_name)
                    break
        except Exception as e:
            print(f"Warning: Could not load foreign keys from tables.json: {e}")
    
    def _get_text_generator(self, col_name: str) -> str:
        """Get SQL expression to generate text based on column name"""
        col_lower = col_name.lower()
        
        # For specific column types, use CASE statements with random selection
        if 'name' in col_lower:
            names = self.text_samples.get('name', ['Alice', 'Bob', 'Charlie', 'Diana'])
            case_parts = [f"WHEN CAST(random() * {len(names)} AS INTEGER) = {i} THEN '{name}'"
                         for i, name in enumerate(names)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        elif 'country' in col_lower:
            countries = self.text_samples.get('country', ['USA', 'UK', 'Canada', 'France'])
            case_parts = [f"WHEN CAST(random() * {len(countries)} AS INTEGER) = {i} THEN '{country}'"
                         for i, country in enumerate(countries)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        elif 'city' in col_lower:
            cities = self.text_samples.get('city', ['New York', 'London', 'Paris', 'Tokyo'])
            case_parts = [f"WHEN CAST(random() * {len(cities)} AS INTEGER) = {i} THEN '{city}'"
                         for i, city in enumerate(cities)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        elif 'location' in col_lower:
            locations = self.text_samples.get('location', ['North', 'South', 'East', 'West'])
            case_parts = [f"WHEN CAST(random() * {len(locations)} AS INTEGER) = {i} THEN '{loc}'"
                         for i, loc in enumerate(locations)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        elif 'theme' in col_lower:
            themes = self.text_samples.get('theme', ['Rock', 'Pop', 'Jazz', 'Classical'])
            case_parts = [f"WHEN CAST(random() * {len(themes)} AS INTEGER) = {i} THEN '{theme}'"
                         for i, theme in enumerate(themes)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        elif 'song' in col_lower:
            songs = self.text_samples.get('song', ['Song A', 'Song B', 'Song C', 'Song D'])
            case_parts = [f"WHEN CAST(random() * {len(songs)} AS INTEGER) = {i} THEN '{song}'"
                         for i, song in enumerate(songs)]
            return f"CASE {' '.join(case_parts)} ELSE 'Unknown' END"
        else:
            # Generate random text using hex(randomblob)
            length = random.randint(8, 16)
            return f"substr(hex(randomblob({length})), 1, {length})"
    
    def _get_column_generator(self, table_name: str, col: Dict, conn: sqlite3.Connection, is_id: bool = False) -> str:
        """Get SQL expression to generate column value"""
        col_name = col['name']
        col_type = col['type']
        
        # Note: ID columns and foreign keys are handled in _generate_data
        # This function only handles regular (non-ID, non-FK) columns
        
        # Generate based on type
        if 'number' in col_type or 'integer' in col_type or 'float' in col_type or 'numeric' in col_type:
            if 'float' in col_type or 'real' in col_type:
                return "random() * 1000"
            else:
                return "CAST(random() * 1000000 AS INTEGER)"
        elif 'text' in col_type or 'varchar' in col_type or 'string' in col_type:
            return self._get_text_generator(col_name)
        elif 'date' in col_type or 'time' in col_type:
            return "date('now', '-' || CAST(random() * 365 AS INTEGER) || ' days')"
        elif 'boolean' in col_type or 'bool' in col_type:
            return "CASE WHEN random() < 0.5 THEN 0 ELSE 1 END"
        else:
            # Default: random text
            length = random.randint(5, 15)
            return f"substr(hex(randomblob({length})), 1, {length})"
    
    def create_database(self, output_path: str):
        """Create SQLite database with tables and data"""
        db_path = Path(output_path) / f"{self.db_name}.db"
        if db_path.exists():
            db_path.unlink()  # Remove existing database
        
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables in order (respecting foreign key dependencies)
        for table_name, columns in self.tables.items():
            self._create_table(cursor, table_name, columns)
        
        # Generate data using recursive CTEs
        for table_name, columns in self.tables.items():
            self._generate_data(cursor, conn, table_name, columns)
        
        conn.commit()
        conn.close()
        
        print(f"Created database: {db_path}")
        return db_path
    
    def _create_table(self, cursor: sqlite3.Cursor, table_name: str, columns: List[Dict]):
        """Create a table with given columns"""
        # Clean identifiers
        clean_table_name = clean_identifier(table_name, prefix="tbl_")
        
        col_defs = []
        for col in columns:
            col_name = clean_identifier(col['name'], prefix="col_")
            
            col_type = col['type']
            
            # Map types to SQLite types
            if 'number' in col_type or 'integer' in col_type or 'int' in col_type:
                sql_type = 'INTEGER'
            elif 'float' in col_type or 'real' in col_type or 'numeric' in col_type:
                sql_type = 'REAL'
            elif 'date' in col_type or 'time' in col_type:
                sql_type = 'TEXT'
            elif 'boolean' in col_type or 'bool' in col_type:
                sql_type = 'INTEGER'
            else:
                sql_type = 'TEXT'
            
            # Quote identifiers to handle special cases
            col_defs.append(f'"{col_name}" {sql_type}')
        
        # Quote table name as well
        create_sql = f'CREATE TABLE IF NOT EXISTS "{clean_table_name}" (\n    ' + \
                    ',\n    '.join(col_defs) + '\n)'
        
        cursor.execute(create_sql)
    
    def _generate_data(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection, 
                      table_name: str, columns: List[Dict], num_rows: int = 1000):
        """Generate data using recursive CTE"""
        clean_table_name = clean_identifier(table_name, prefix="tbl_")
        
        # Build column list and value generators
        col_names = []
        base_generators = []  # For base case (n=1)
        recursive_generators = []  # For recursive case (n+1)
        id_column_indices = []  # Track which columns are IDs
        
        for idx, col in enumerate(columns):
            col_name = clean_identifier(col['name'], prefix="col_")
            col_names.append(col_name)
            
            # Check if this is a foreign key
            is_fk = False
            if table_name in self.foreign_keys:
                for fk in self.foreign_keys[table_name]:
                    if fk['child_col'].lower() == col['name'].lower():
                        parent_table = fk['parent_table']
                        parent_col = fk['parent_col']
                        clean_parent_table = clean_identifier(parent_table, prefix="tbl_")
                        clean_parent_col = clean_identifier(parent_col, prefix="col_")
                        
                        # Check if parent table has data
                        cursor.execute(f'SELECT COUNT(*) FROM "{clean_parent_table}"')
                        parent_count = cursor.fetchone()[0]
                        
                        if parent_count > 0:
                            # Use subquery to get random value from parent
                            fk_expr = f'(SELECT "{clean_parent_col}" FROM "{clean_parent_table}" ORDER BY RANDOM() LIMIT 1)'
                            base_generators.append(fk_expr)
                            recursive_generators.append(fk_expr)  # Same for recursive case
                        else:
                            # Parent empty, use NULL
                            base_generators.append("NULL")
                            recursive_generators.append("NULL")
                        is_fk = True
                        break
            
            if not is_fk:
                col_name_lower = col['name'].lower()
                is_id = col.get('is_id', False) or col_name_lower in [pk.lower() for pk in self.primary_keys.get(table_name, [])]
                
                if is_id:
                    # ID column: use n in base case, n in recursive case
                    base_generators.append("1")  # Start with 1
                    recursive_generators.append("n + 1")  # Increment in recursive case
                    id_column_indices.append(idx)
                else:
                    # Regular column: get generator expression
                    generator = self._get_column_generator(table_name, col, conn, is_id=False)
                    base_generators.append(generator)
                    recursive_generators.append(generator)  # Regenerate each time
        
        # Create recursive CTE to generate N rows
        # Quote all identifiers
        quoted_col_names = [f'"{name}"' for name in col_names]
        cte_sql = f"""
        WITH RECURSIVE generate_rows(n, {', '.join(quoted_col_names)}) AS (
            SELECT 1, {', '.join(base_generators)}
            UNION ALL
            SELECT n + 1, {', '.join(recursive_generators)}
            FROM generate_rows
            WHERE n < {num_rows}
        )
        INSERT INTO "{clean_table_name}" ({', '.join(quoted_col_names)})
        SELECT {', '.join(quoted_col_names)} FROM generate_rows;
        """
        
        try:
            cursor.execute(cte_sql)
            conn.commit()
            print(f"  Generated {num_rows} rows for table: {clean_table_name} (recursive CTE)")
        except sqlite3.Error as e:
            print(f"  Error with recursive CTE for {clean_table_name}: {e}")
            print(f"  Falling back to batch INSERT method...")
            # Fallback: batch INSERT
            self._generate_data_fallback(cursor, conn, clean_table_name, columns, num_rows)
    
    def _generate_data_fallback(self, cursor: sqlite3.Cursor, conn: sqlite3.Connection,
                               table_name: str, columns: List[Dict], num_rows: int):
        """Fallback method using batch INSERT"""
        clean_table_name = clean_identifier(table_name, prefix="tbl_")
        col_names = [clean_identifier(col['name'], prefix="col_") for col in columns]
        
        # Prepare batch insert
        batch_size = 100
        placeholders = ','.join(['?' for _ in col_names])
        quoted_col_names = [f'"{name}"' for name in col_names]
        insert_sql = f'INSERT INTO "{clean_table_name}" ({", ".join(quoted_col_names)}) VALUES ({placeholders})'
        
        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(batch_start + batch_size, num_rows)
            batch_values = []
            
            for i in range(batch_start, batch_end):
                row_values = []
                for col in columns:
                    col_name = col['name']
                    col_type = col['type']
                    is_id = col.get('is_id', False) or col_name.lower() in [pk.lower() for pk in self.primary_keys.get(table_name, [])]
                    
                    # Check if foreign key
                    is_fk = False
                    if table_name in self.foreign_keys:
                        for fk in self.foreign_keys[table_name]:
                            if fk['child_col'].lower() == col_name.lower():
                                parent_table = fk['parent_table']
                                parent_col = fk['parent_col']
                                clean_parent_table = clean_identifier(parent_table, prefix="tbl_")
                                clean_parent_col = clean_identifier(parent_col, prefix="col_")
                                
                                cursor.execute(f'SELECT "{clean_parent_col}" FROM "{clean_parent_table}" ORDER BY RANDOM() LIMIT 1')
                                val = cursor.fetchone()
                                row_values.append(val[0] if val else None)
                                is_fk = True
                                break
                    
                    if not is_fk:
                        # ID column
                        if is_id:
                            row_values.append(i + 1)
                        # Regular column - generate value
                        else:
                            generator = self._get_column_generator(table_name, col, conn, is_id=False)
                            cursor.execute(f"SELECT ({generator})")
                            val = cursor.fetchone()[0]
                            row_values.append(val)
                
                batch_values.append(row_values)
            
            cursor.executemany(insert_sql, batch_values)
        
        conn.commit()
        print(f"  Generated {num_rows} rows for table: {table_name} (fallback method)")


def main():
    # Configuration
    val_json_path = "data/processed/val.json"
    tables_json_path = "data/spider/tables.json"
    output_dir = Path("data/generated_databases")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load validation data
    print(f"Loading schemas from {val_json_path}...")
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    # Extract unique schemas
    unique_schemas = OrderedDict()
    for example in val_data:
        schema_text = example.get('schema', '')
        if schema_text and schema_text not in unique_schemas:
            unique_schemas[schema_text] = example.get('db_id', 'unknown')
    
    print(f"Found {len(unique_schemas)} unique schemas")
    
    # Process each unique schema
    parser = SchemaParser()
    
    for idx, (schema_text, db_id) in enumerate(unique_schemas.items(), 1):
        print(f"\n[{idx}/{len(unique_schemas)}] Processing schema: {db_id}")
        
        # Parse schema
        schema_info = parser.parse(schema_text)
        if not schema_info['tables']:
            print(f"  Warning: No tables found in schema")
            continue
        
        print(f"  Found {len(schema_info['tables'])} tables")
        for table_name in schema_info['tables']:
            print(f"    - {table_name}: {len(schema_info['tables'][table_name])} columns")
        
        # Generate database
        generator = DataGenerator(schema_info, tables_json_path)
        db_path = generator.create_database(output_dir)
        
        # Verify
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        for table_name in schema_info['tables']:
            clean_name = clean_identifier(table_name, prefix="tbl_")
            cursor.execute(f'SELECT COUNT(*) FROM "{clean_name}"')
            count = cursor.fetchone()[0]
            print(f"    ✓ {clean_name}: {count} rows")
        conn.close()
    
    print(f"\n✓ Successfully generated {len(unique_schemas)} databases in {output_dir}")


if __name__ == "__main__":
    main()

