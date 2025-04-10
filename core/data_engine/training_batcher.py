import numpy as np
from collections import deque
from core.utilities import advanced_logger

class DynamicBatcher:
    def __init__(self, max_batch_size=32):
        self.logger = advanced_logger.AiraLogger(self.__class__.__name__)
        self.buffer = deque(maxlen=10000)
        self.max_batch_size = max_batch_size
        self.batch_strategy = 'dynamic'
        
    def add_interaction(self, interaction):
        self.buffer.append(interaction)
        self._auto_adjust_batch_size()
        
    def get_batch(self):
        if len(self.buffer) < self.max_batch_size:
            return None
            
        batch = list(self.buffer)[:self.max_batch_size]
        self.buffer = deque(list(self.buffer)[self.max_batch_size:], maxlen=10000)
        return self._process_batch(batch)
    
    def _process_batch(self, raw_batch):
        # Procesamiento avanzado con balanceo de clases
        tokenized = [self._tokenize(item) for item in raw_batch]
        return self._pad_sequences(tokenized)
    
    def _tokenize(self, text):
        return self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
    
    def _pad_sequences(self, sequences):
        # Padding inteligente basado en longitudes
        lengths = [len(seq['input_ids']) for seq in sequences]
        max_len = max(lengths)
        
        padded_batch = {
            'input_ids': torch.zeros((len(sequences), max_len), dtype=torch.long),
            'attention_mask': torch.zeros((len(sequences), max_len), dtype=torch.long)
        }
        
        for i, seq in enumerate(sequences):
            padded_batch['input_ids'][i, :len(seq['input_ids'])] = seq['input_ids']
            padded_batch['attention_mask'][i, :len(seq['attention_mask'])] = seq['attention_mask']
            
        return padded_batch
    
    def _auto_adjust_batch_size(self):
        # Algoritmo de ajuste automático de batch size
        if len(self.buffer) > 5000 and self.max_batch_size < 64:
            self.max_batch_size *= 2
            self.logger.info(f"Auto-adjusted batch size to {self.max_batch_size}")