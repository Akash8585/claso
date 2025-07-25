import re
from collections import Counter
from typing import List, Dict
import pickle

class CodeTokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '<pad>': 0, '<unk>': 1, '<sos>': 2, '<eos>': 3,
            '<code>': 4, '<comment>': 5, '<indent>': 6, '<dedent>': 7
        }
    def preprocess_code(self, code: str) -> str:
        lines = code.split('\n')
        processed_lines = []
        for line in lines:
            indent_level = len(line) - len(line.lstrip())
            if indent_level > 0:
                processed_lines.append('<indent>' * (indent_level // 4) + line.strip())
            else:
                processed_lines.append(line.strip())
        return '\n'.join(processed_lines)
    def tokenize(self, text: str) -> List[str]:
        patterns = {
            r'\b\d+\b': '<NUM>',
            r'\b[a-zA-Z_][a-zA-Z0-9_]*\b': lambda m: m.group(),
            r'[+\-*/=<>!&|]+': lambda m: m.group(),
            r'[(){}[\];,.]': lambda m: m.group(),
        }
        tokens = []
        for match in re.finditer('|'.join(patterns.keys()), text):
            tokens.append(match.group())
        return tokens
    def build_vocabulary(self, texts: List[str]):
        token_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        most_common = token_counts.most_common(self.vocab_size - len(self.special_tokens))
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        for token, _ in most_common:
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(token, self.token_to_id['<unk>']) for token in tokens]
    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.id_to_token.get(i, '<unk>') for i in ids])
    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'token_to_id': self.token_to_id,
                'id_to_token': self.id_to_token,
                'vocab_size': self.vocab_size
            }, f)
    def load(self, path: str):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.token_to_id = data['token_to_id']
            self.id_to_token = data['id_to_token']
            self.vocab_size = data['vocab_size'] 