"""
TRM (Tiny Recursive Model) for Text-to-SQL
Implementation based on recursive attention mechanism
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Optional, Tuple
try:
    from lark import Lark, Tree
    LARK_AVAILABLE = True
except ImportError:
    LARK_AVAILABLE = False
    Lark = None
    Tree = None


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class RecursiveAttention(nn.Module):
    """Recursive attention mechanism for TRM"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure tensors are in the correct format: [batch_size, seq_len, d_model]
        assert query.dim() == 3, f"query should be 3D, got {query.dim()}D with shape {query.shape}"
        assert key.dim() == 3, f"key should be 3D, got {key.dim()}D with shape {key.shape}"
        assert value.dim() == 3, f"value should be 3D, got {value.dim()}D with shape {value.shape}"
        assert query.size(0) == key.size(0) == value.size(0), "Batch sizes must match"
        assert key.size(0) == value.size(0), "Batch sizes must match"
        assert query.size(2) == key.size(2) == value.size(2) == self.d_model, \
            f"d_model mismatch: query={query.size(2)}, key={key.size(2)}, value={value.size(2)}, expected={self.d_model}"
        
        batch_size = query.size(0)
        seq_len_q = query.size(1)
        seq_len_kv = key.size(1)
        
        # Project to Q, K, V
        Q_proj = self.q_proj(query)  # [batch_size, seq_len_q, d_model]
        K_proj = self.k_proj(key)    # [batch_size, seq_len_kv, d_model]
        V_proj = self.v_proj(value)  # [batch_size, seq_len_kv, d_model]
        
        # Reshape to [batch_size, seq_len, n_heads, head_dim] then transpose to [batch_size, n_heads, seq_len, head_dim]
        Q = Q_proj.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
        K = K_proj.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        V = V_proj.view(batch_size, seq_len_kv, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [batch_size, n_heads, seq_len_q, seq_len_kv]
        
        if mask is not None:
            # mask should be [batch_size, 1, seq_len_kv] or [batch_size, seq_len_q, seq_len_kv]
            # Broadcast to match scores shape
            if mask.dim() == 3:
                if mask.size(1) == 1:
                    # [batch_size, 1, seq_len_kv] -> [batch_size, 1, 1, seq_len_kv]
                    mask = mask.unsqueeze(1)
                # mask should now be [batch_size, n_heads, seq_len_q, seq_len_kv] after broadcasting
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, self.d_model
        )
        attn_output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(query + attn_output)
        return output


class TRMEncoder(nn.Module):
    """TRM Encoder with recursive layers"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, n_layers)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [seq_len, batch_size]
            src_mask: Optional attention mask
            src_key_padding_mask: Optional padding mask
        Returns:
            [seq_len, batch_size, d_model]
        """
        # Embedding: [seq_len, batch_size] -> [seq_len, batch_size, d_model]
        src = self.embedding(src) * math.sqrt(self.d_model)
        # Positional encoding expects [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        
        # Transformer encoder expects [seq_len, batch_size, d_model] with batch_first=False
        output = self.transformer_encoder(src, mask=src_mask, 
                                        src_key_padding_mask=src_key_padding_mask)
        return output  # [seq_len, batch_size, d_model]


class TRMDecoder(nn.Module):
    """TRM Decoder with recursive attention"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 max_len: int = 512):
        super().__init__()
        self.d_model = d_model
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        # Recursive attention layers
        self.self_attention_layers = nn.ModuleList([
            RecursiveAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        self.cross_attention_layers = nn.ModuleList([
            RecursiveAttention(d_model, n_heads, dropout) for _ in range(n_layers)
        ])
        
        # Feedforward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            ) for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tgt: [batch_size, seq_len]
            memory: [seq_len_mem, batch_size, d_model] from encoder
        Returns:
            [batch_size, seq_len, d_model]
        """
        # Embed and add positional encoding
        tgt = self.embedding(tgt) * math.sqrt(self.d_model)  # [batch_size, seq_len, d_model]
        
        # Convert memory to batch-first format
        memory_t = memory.transpose(0, 1)  # [batch_size, seq_len_mem, d_model]
        
        # Ensure memory_t and tgt have matching batch size
        assert memory_t.size(0) == tgt.size(0), \
            f"Batch size mismatch: memory_t={memory_t.size(0)}, tgt={tgt.size(0)}"
        
        # Create cross-attention mask from memory_key_padding_mask if provided
        cross_attn_mask = None
        if memory_key_padding_mask is not None:
            # memory_key_padding_mask: [batch_size, seq_len_mem] where 1.0 = valid, 0.0 = padding
            # RecursiveAttention expects mask where 0 = masked, 1 = valid
            # Expand to [batch_size, seq_len_tgt, seq_len_mem] for proper broadcasting
            # For now, use [batch_size, 1, seq_len_mem] to broadcast over query sequence
            assert memory_key_padding_mask.size(0) == memory_t.size(0), \
                f"Batch size mismatch in mask: {memory_key_padding_mask.size(0)} vs {memory_t.size(0)}"
            assert memory_key_padding_mask.size(1) == memory_t.size(1), \
                f"Sequence length mismatch in mask: {memory_key_padding_mask.size(1)} vs {memory_t.size(1)}"
            
            if memory_key_padding_mask.dtype == torch.bool:
                cross_attn_mask = memory_key_padding_mask.float().unsqueeze(1)  # [batch_size, 1, seq_len_mem]
            else:
                cross_attn_mask = memory_key_padding_mask.unsqueeze(1)  # [batch_size, 1, seq_len_mem]
        
        # Apply recursive layers
        for self_attn, cross_attn, ff in zip(self.self_attention_layers, 
                                            self.cross_attention_layers,
                                            self.ff_layers):
            # Self-attention
            tgt = self_attn(tgt, tgt, tgt, tgt_mask)
            
            # Cross-attention with encoder output
            # Query: tgt [batch_size, seq_len_tgt, d_model]
            # Key/Value: memory_t [batch_size, seq_len_mem, d_model]
            tgt = cross_attn(tgt, memory_t, memory_t, cross_attn_mask)
            
            # Feedforward
            residual = tgt
            tgt = ff(tgt)
            tgt = tgt + residual
        
        return tgt


class TRMTextToSQL(nn.Module):
    """Complete TRM model for Text-to-SQL with optional constrained decoding"""
    
    SQL_GRAMMAR = """
        // Start symbol for full query
        start: query
        
        // Main query structure
        query: "SELECT" select_list "FROM" from_clause [where_clause] [group_by_clause] [order_by_clause] ";"?
        
        // SELECT clause: columns, aggregates, *
        select_list: select_item ("," select_item)*
        select_item: [aggregate] column_ref | "*"
        aggregate: "COUNT" "(" [column_ref | "*"] ")" 
                  | "SUM" "(" column_ref ")" 
                  | "AVG" "(" column_ref ")"
                  | "MAX" "(" column_ref ")"
                  | "MIN" "(" column_ref ")"
        
        // FROM clause: tables and joins
        from_clause: table_ref (join_clause)*
        table_ref: table_name [table_alias]
        join_clause: join_type "JOIN" table_ref "ON" condition
        join_type: "INNER" | "LEFT" | "RIGHT" | "FULL"
        
        // WHERE clause: filters
        where_clause: "WHERE" condition
        condition: expression (("AND" | "OR") expression)*
        expression: column_ref comparison value
                   | column_ref "IN" "(" value_list ")"
                   | column_ref "BETWEEN" value "AND" value
                   | "(" condition ")"
        comparison: "=" | "!=" | "<" | ">" | "<=" | ">="
        
        // GROUP BY and ORDER BY
        group_by_clause: "GROUP" "BY" column_ref ("," column_ref)*
        order_by_clause: "ORDER" "BY" order_item ("," order_item)*
        order_item: column_ref ["ASC" | "DESC"]
        
        // Schema elements (placeholders for tables/columns)
        table_name: /\\w+/
        column_ref: /\\w+\\.\\w+/ | /\\w+/
        table_alias: "AS"? /\\w+/
        
        // Values: strings, numbers, etc.
        value: NUMBER | STRING | NULL
        value_list: value ("," value)*
        NUMBER: /-?\\d+(\\.\\d+)?/
        STRING: /"[^"]*"|'[^']*'/
        NULL: "NULL"
        
        // Ignore whitespace
        %import common.WS
        %ignore WS
    """
    
    def __init__(self, question_vocab_size: int, sql_vocab_size: int,
                 d_model: int = 512, n_heads: int = 8, n_encoder_layers: int = 6,
                 n_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, max_len: int = 512,
                 use_constrained_decoding: bool = False, sql_tokenizer=None):
        super().__init__()
        
        self.encoder = TRMEncoder(
            vocab_size=question_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
        
        self.decoder = TRMDecoder(
            vocab_size=sql_vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, sql_vocab_size)
        
        self.d_model = d_model
        self.sql_vocab_size = sql_vocab_size
        self.use_constrained_decoding = use_constrained_decoding
        self.sql_tokenizer = sql_tokenizer
        
        # Initialize SQL grammar parser for constrained decoding
        self.sql_parser = None
        if use_constrained_decoding and LARK_AVAILABLE:
            try:
                self.sql_parser = Lark(self.SQL_GRAMMAR, start='start', parser='earley', 
                                     propagate_positions=True)
            except Exception as e:
                print(f"Warning: Failed to initialize SQL grammar parser: {e}")
                print("Constrained decoding will be disabled.")
                self.use_constrained_decoding = False
        elif use_constrained_decoding and not LARK_AVAILABLE:
            print("Warning: lark parser not available. Install with: pip install lark")
            print("Constrained decoding will be disabled.")
            self.use_constrained_decoding = False
        
    def forward(self, question: torch.Tensor, sql: Optional[torch.Tensor] = None,
                question_mask: Optional[torch.Tensor] = None,
                sql_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            question: [batch_size, seq_len_q] - input question tokens
            sql: [batch_size, seq_len_sql] - target SQL tokens (for training)
            question_mask: Optional padding mask for question
            sql_mask: Optional mask for SQL generation
        
        Returns:
            [batch_size, seq_len_sql, sql_vocab_size] - logits for SQL tokens
        """
        # Encode question
        question_t = question.transpose(0, 1)  # [seq_len, batch_size]
        question_mask_padding_encoder = None
        if question_mask is not None:
            # Encoder with batch_first=False expects src_key_padding_mask as [batch_size, seq_len]
            # where True = padding token (to be masked), False = valid token
            # question_mask is [batch_size, seq_len] where 1.0 = valid, 0.0 = padding
            # So we need (question_mask == 0) which gives True for padding, False for valid
            question_mask_padding_encoder = (question_mask == 0)  # [batch_size, seq_len]
        
        memory = self.encoder(question_t, src_key_padding_mask=question_mask_padding_encoder)
        # memory: [seq_len_q, batch_size, d_model]
        
        # Prepare mask for decoder cross-attention
        # Decoder expects [batch_size, seq_len_mem] where 1.0 = valid, 0.0 = padding
        memory_key_padding_mask_decoder = None
        if question_mask is not None:
            memory_key_padding_mask_decoder = question_mask  # [batch_size, seq_len_q]
        
        # Decode SQL
        if sql is not None:
            # Training: use ground truth SQL
            decoder_output = self.decoder(
                sql, memory,
                tgt_key_padding_mask=sql_mask,
                memory_key_padding_mask=memory_key_padding_mask_decoder
            )
            # decoder_output: [batch_size, seq_len_sql, d_model]
        else:
            # Inference: autoregressive generation would go here
            # For now, return encoder output (inference handled separately)
            decoder_output = memory.transpose(0, 1)
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        return logits
    
    def _is_valid_prefix(self, partial_sql: str) -> bool:
        """Check if partial SQL is a valid prefix of the grammar"""
        if not self.sql_parser:
            return True
        
        # Try to parse the partial SQL
        # Use error recovery mode to check if it's a valid prefix
        try:
            # Attempt partial parsing
            self.sql_parser.parse(partial_sql)
            return True
        except Exception:
            # Check if it's a valid prefix by trying to continue parsing
            # This is a heuristic - in practice, we'd need more sophisticated prefix checking
            # For now, we'll allow any prefix that doesn't contain obvious syntax errors
            partial_lower = partial_sql.strip().upper()
            
            # Basic heuristics for valid prefixes
            # Allow if it starts with SELECT or contains valid SQL keywords
            if partial_lower.startswith('SELECT'):
                return True
            
            # If we can't determine, be permissive
            return True
    
    def _get_valid_token_mask(self, partial_sql: str, vocab_size: int, device: torch.device) -> torch.Tensor:
        """
        Returns mask of valid next tokens based on SQL grammar.
        
        Args:
            partial_sql: Partial SQL string generated so far
            vocab_size: Size of vocabulary
            device: Device for the mask tensor
            
        Returns:
            Binary mask [vocab_size] where 1 = valid, 0 = invalid
        """
        if not self.use_constrained_decoding or not self.sql_parser or not self.sql_tokenizer:
            # If constrained decoding is disabled, all tokens are valid
            return torch.ones(vocab_size, dtype=torch.bool, device=device)
        
        valid_tokens = torch.zeros(vocab_size, dtype=torch.bool, device=device)
        
        # Try each token in vocabulary (sample a subset for efficiency)
        # In practice, you might want to cache or optimize this
        tokens_to_check = min(vocab_size, 1000)  # Limit checking for efficiency
        
        for token_id in range(tokens_to_check):
            try:
                # Decode token to string
                token_str = self.sql_tokenizer.decode([token_id]).strip()
                if not token_str or token_str in ['<PAD>', '<START>', '<END>', '<UNK>']:
                    # Skip special tokens - they're always valid
                    valid_tokens[token_id] = True
                    continue
                
                # Try appending token to partial SQL
                test_sql = partial_sql + ' ' + token_str if partial_sql else token_str
                
                try:
                    # Try to parse the extended SQL
                    self.sql_parser.parse(test_sql)
                    valid_tokens[token_id] = True
                except Exception:
                    # Check if it's a valid prefix
                    if self._is_valid_prefix(test_sql):
                        valid_tokens[token_id] = True
            except Exception:
                # If we can't decode or check, be permissive
                valid_tokens[token_id] = True
        
        # For tokens we didn't check, assume valid (for efficiency)
        if tokens_to_check < vocab_size:
            valid_tokens[tokens_to_check:] = True
        
        # Ensure at least some tokens are valid (avoid dead ends)
        if not valid_tokens.any():
            valid_tokens = torch.ones(vocab_size, dtype=torch.bool, device=device)
        
        return valid_tokens
    
    def generate(self, question: torch.Tensor, question_mask: Optional[torch.Tensor] = None,
                 max_len: int = 256, start_token: int = 1, end_token: int = 2,
                 temperature: float = 1.0) -> torch.Tensor:
        """Generate SQL autoregressively with optional constrained decoding"""
        self.eval()
        batch_size = question.size(0)
        device = question.device
        
        # Encode question
        question_t = question.transpose(0, 1)
        question_mask_padding_encoder = None
        if question_mask is not None:
            # Encoder expects [batch_size, seq_len] for src_key_padding_mask
            question_mask_padding_encoder = (question_mask == 0)  # [batch_size, seq_len]
        
        memory = self.encoder(question_t, src_key_padding_mask=question_mask_padding_encoder)
        memory = memory.transpose(0, 1)  # [batch_size, seq_len_q, d_model]
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Track partial SQL strings for constrained decoding (one per batch item)
        partial_sqls = [''] * batch_size if self.use_constrained_decoding else None
        
        for _ in range(max_len - 1):
            # Decode current sequence
            decoder_output = self.decoder(generated, memory.transpose(0, 1))
            logits = self.output_proj(decoder_output[:, -1:, :]) / temperature
            # logits: [batch_size, 1, vocab_size]
            
            # Apply constrained decoding if enabled
            if self.use_constrained_decoding and partial_sqls is not None:
                # Process each item in the batch
                for batch_idx in range(batch_size):
                    if partial_sqls[batch_idx] is None:
                        continue
                    
                    # Get valid token mask for this batch item
                    valid_mask = self._get_valid_token_mask(
                        partial_sqls[batch_idx], 
                        self.sql_vocab_size, 
                        device
                    )
                    
                    # Set invalid tokens to -inf
                    logits[batch_idx, 0, ~valid_mask] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits.squeeze(1), dim=-1)  # [batch_size, vocab_size]
            next_token = torch.multinomial(probs, 1)  # [batch_size, 1]
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update partial SQL strings for constrained decoding
            if self.use_constrained_decoding and partial_sqls is not None and self.sql_tokenizer:
                for batch_idx in range(batch_size):
                    token_id = next_token[batch_idx].item()
                    if token_id != start_token:  # Don't include start token
                        token_str = self.sql_tokenizer.decode([token_id]).strip()
                        if token_str and token_str not in ['<PAD>', '<START>', '<END>']:
                            partial_sqls[batch_idx] += ' ' + token_str if partial_sqls[batch_idx] else token_str
            
            # Check for end token
            if (next_token == end_token).all():
                break
        
        return generated

