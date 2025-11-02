"""
Simple tokenizer for Text-to-SQL
"""
from collections import defaultdict
import re


class SimpleTokenizer:
    """Simple tokenizer for demonstration (replace with proper tokenizer for production)"""
    
    def __init__(self, vocab_file=None):
        self.vocab = defaultdict(lambda: len(self.vocab))
        self.vocab['<PAD>'] = 0
        self.vocab['<START>'] = 1
        self.vocab['<END>'] = 2
        self.vocab['<UNK>'] = 3
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self._build_vocab = True
    
    def _tokenize(self, text):
        """Simple word-level tokenization"""
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens
    
    def encode(self, text, max_length=512, truncation=True, padding='max_length'):
        """Encode text to token IDs"""
        tokens = self._tokenize(text)
        
        if truncation and len(tokens) > max_length - 2:
            tokens = tokens[:max_length - 2]
        
        token_ids = [self.vocab['<START>']]
        for token in tokens:
            if token not in self.vocab:
                if self._build_vocab:
                    self.vocab[token] = len(self.vocab)
                else:
                    token = '<UNK>'
            token_ids.append(self.vocab.get(token, self.vocab['<UNK>']))
        token_ids.append(self.vocab['<END>'])
        
        # Padding
        if padding == 'max_length':
            while len(token_ids) < max_length:
                token_ids.append(self.vocab['<PAD>'])
        
        return token_ids
    
    def decode(self, token_ids):
        """Decode token IDs to text"""
        tokens = [self.inv_vocab.get(tid, '<UNK>') for tid in token_ids]
        return ' '.join(tokens)
    
    def get_vocab_size(self):
        return len(self.vocab)

