import torch
from typing import List
from collections import Counter
from torchtext.data.utils import get_tokenizer

class SimpleTokenizer:
    def __init__(self, min_freq: int = 2, max_tokens: int = 20000):
        self.word2idx = {}
        self.idx2word = {}
        self.min_freq = min_freq
        self.max_tokens = max_tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.base_tokenizer = get_tokenizer("basic_english")
        
    def fit(self, texts: List[str]):
        word_counts = Counter()
        for text in texts:
            words = self.base_tokenizer(text.lower())
            word_counts.update(words)
        
        # Select top k tokens
        vocab = [self.pad_token, self.unk_token]
        vocab.extend([word for word, count in word_counts.most_common(self.max_tokens - 2)
                     if count >= self.min_freq])
        
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        words = self.base_tokenizer(text.lower())
        tokens = [self.word2idx.get(word, self.word2idx[self.unk_token]) 
                 for word in words]
        
        if len(tokens) < max_len:
            tokens = tokens + [self.word2idx[self.pad_token]] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        
        return tokens
    
    def vocab_size(self) -> int:
        return len(self.word2idx)