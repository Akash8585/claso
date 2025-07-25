#!/usr/bin/env python3
"""
Advanced data processing and augmentation for production training
"""

import json
import re
import random
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os

@dataclass
class ProcessedPair:
    code: str
    comment: str
    language: str
    quality_score: float
    augmented: bool = False

class DataProcessor:
    def __init__(self):
        self.python_keywords = [
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'yield', 'lambda', 'with', 'as', 'pass',
            'break', 'continue', 'global', 'nonlocal', 'assert', 'del', 'raise'
        ]
        
        self.js_keywords = [
            'function', 'class', 'if', 'else', 'for', 'while', 'try', 'catch',
            'import', 'export', 'return', 'yield', 'async', 'await', 'const',
            'let', 'var', 'break', 'continue', 'throw', 'switch', 'case'
        ]
    
    def load_enhanced_data(self, file_path: str) -> List[ProcessedPair]:
        """Load enhanced data from JSON"""
        if not os.path.exists(file_path):
            print(f"⚠️  File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        pairs = []
        for item in data:
            pair = ProcessedPair(
                code=item['code'],
                comment=item['comment'],
                language=item['language'],
                quality_score=item.get('quality_score', 0.5)
            )
            pairs.append(pair)
        
        return pairs
    
    def clean_code(self, code: str, language: str) -> str:
        """Clean and normalize code"""
        # Remove excessive whitespace
        lines = [line.rstrip() for line in code.split('\n')]
        
        # Remove empty lines at start and end
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        
        # Normalize indentation
        if lines:
            min_indent = min(len(line) - len(line.lstrip()) 
                           for line in lines if line.strip())
            lines = [line[min_indent:] if line.strip() else line 
                    for line in lines]
        
        return '\n'.join(lines)
    
    def clean_comment(self, comment: str, language: str) -> str:
        """Clean and normalize comments"""
        # Remove comment markers
        comment = re.sub(r'^/\*\*\s*', '', comment)
        comment = re.sub(r'\s*\*/$', '', comment)
        comment = re.sub(r'^\s*\*\s?', '', comment, flags=re.MULTILINE)
        
        # Clean up whitespace
        lines = [line.strip() for line in comment.split('\n')]
        lines = [line for line in lines if line]  # Remove empty lines
        
        return '\n'.join(lines)
    
    def augment_python_code(self, code: str, comment: str) -> List[Tuple[str, str]]:
        """Create variations of Python code-comment pairs"""
        variations = [(code, comment)]  # Original
        
        # Variation 1: Change variable names
        var_mapping = {
            'data': 'items', 'result': 'output', 'value': 'val',
            'number': 'num', 'string': 'text', 'list': 'arr',
            'dict': 'mapping', 'key': 'k', 'item': 'element'
        }
        
        modified_code = code
        for old_var, new_var in var_mapping.items():
            if f' {old_var}' in modified_code or f'({old_var}' in modified_code:
                modified_code = re.sub(rf'\b{old_var}\b', new_var, modified_code)
        
        if modified_code != code:
            variations.append((modified_code, comment))
        
        # Variation 2: Add type hints (if not present)
        if 'def ' in code and '->' not in code and ':' in code:
            # Simple type hint addition
            type_hint_code = re.sub(
                r'def (\w+)\(([^)]*)\):',
                r'def \1(\2) -> None:',
                code
            )
            if type_hint_code != code:
                variations.append((type_hint_code, comment))
        
        # Variation 3: Different formatting styles
        if len(code.split('\n')) > 1:
            # More compact formatting
            compact_code = re.sub(r'\n\s*\n', '\n', code)  # Remove blank lines
            if compact_code != code:
                variations.append((compact_code, comment))
        
        return variations
    
    def augment_javascript_code(self, code: str, comment: str) -> List[Tuple[str, str]]:
        """Create variations of JavaScript code-comment pairs"""
        variations = [(code, comment)]  # Original
        
        # Variation 1: Convert function declaration to arrow function
        if 'function ' in code and '=>' not in code:
            arrow_code = re.sub(
                r'function\s+(\w+)\s*\(([^)]*)\)\s*{',
                r'const \1 = (\2) => {',
                code
            )
            if arrow_code != code:
                variations.append((arrow_code, comment))
        
        # Variation 2: Convert var to const/let
        if 'var ' in code:
            const_code = code.replace('var ', 'const ')
            variations.append((const_code, comment))
        
        # Variation 3: Add async/await if applicable
        if 'Promise' in code and 'async' not in code:
            async_code = re.sub(r'function\s+(\w+)', r'async function \1', code)
            if async_code != code:
                variations.append((async_code, comment))
        
        return variations
    
    def augment_comments(self, comment: str, language: str) -> List[str]:
        """Create variations of comments"""
        variations = [comment]
        
        # Variation 1: Different documentation styles
        if language == 'python':
            # Convert to different docstring styles
            if '"""' not in comment:
                docstring_comment = f'"""\n{comment}\n"""'
                variations.append(docstring_comment)
            
            # Add parameter documentation if missing
            if 'Args:' not in comment and 'Parameters:' not in comment:
                enhanced_comment = f"{comment}\n\nArgs:\n    param: Input parameter"
                variations.append(enhanced_comment)
        
        elif language == 'javascript':
            # Convert to JSDoc format if not already
            if '/**' not in comment:
                jsdoc_comment = f"/**\n * {comment}\n */"
                variations.append(jsdoc_comment)
            
            # Add @param and @returns if missing
            if '@param' not in comment:
                enhanced_comment = comment.replace('*/', ' * @param {*} param - Input parameter\n */')
                variations.append(enhanced_comment)
        
        return variations
    
    def balance_dataset(self, pairs: List[ProcessedPair]) -> List[ProcessedPair]:
        """Balance dataset between languages and quality levels"""
        python_pairs = [p for p in pairs if p.language == 'python']
        js_pairs = [p for p in pairs if p.language == 'javascript']
        
        print(f"📊 Before balancing: Python={len(python_pairs)}, JS={len(js_pairs)}")
        
        # Balance languages (aim for 60% Python, 40% JavaScript)
        target_python = int(len(pairs) * 0.6)
        target_js = int(len(pairs) * 0.4)
        
        # Sort by quality and take best ones
        python_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        js_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        
        balanced_pairs = (
            python_pairs[:target_python] + 
            js_pairs[:target_js]
        )
        
        print(f"📊 After balancing: Python={len([p for p in balanced_pairs if p.language == 'python'])}, JS={len([p for p in balanced_pairs if p.language == 'javascript'])}")
        
        return balanced_pairs
    
    def process_and_augment(self, input_files: List[str], output_file: str, target_size: int = 8000):
        """Main processing and augmentation pipeline"""
        print("🔄 Starting data processing and augmentation...")
        
        # Load all data
        all_pairs = []
        for file_path in input_files:
            pairs = self.load_enhanced_data(file_path)
            all_pairs.extend(pairs)
        
        print(f"📊 Loaded {len(all_pairs)} pairs from {len(input_files)} files")
        
        # Clean and process
        processed_pairs = []
        for pair in all_pairs:
            clean_code = self.clean_code(pair.code, pair.language)
            clean_comment = self.clean_comment(pair.comment, pair.language)
            
            # Skip if too short after cleaning
            if len(clean_code.strip()) < 20 or len(clean_comment.strip()) < 15:
                continue
            
            processed_pair = ProcessedPair(
                code=clean_code,
                comment=clean_comment,
                language=pair.language,
                quality_score=pair.quality_score
            )
            processed_pairs.append(processed_pair)
        
        print(f"📊 After cleaning: {len(processed_pairs)} pairs")
        
        # Balance dataset
        balanced_pairs = self.balance_dataset(processed_pairs)
        
        # Augment data if we need more
        if len(balanced_pairs) < target_size:
            print(f"🔄 Augmenting data to reach target size of {target_size}...")
            
            augmented_pairs = []
            for pair in balanced_pairs:
                # Add original
                augmented_pairs.append(pair)
                
                # Add variations
                if pair.language == 'python':
                    variations = self.augment_python_code(pair.code, pair.comment)
                else:
                    variations = self.augment_javascript_code(pair.code, pair.comment)
                
                for var_code, var_comment in variations[1:]:  # Skip original
                    aug_pair = ProcessedPair(
                        code=var_code,
                        comment=var_comment,
                        language=pair.language,
                        quality_score=pair.quality_score * 0.9,  # Slightly lower score
                        augmented=True
                    )
                    augmented_pairs.append(aug_pair)
                
                # Stop if we reach target
                if len(augmented_pairs) >= target_size:
                    break
            
            balanced_pairs = augmented_pairs[:target_size]
        
        # Final shuffle
        random.shuffle(balanced_pairs)
        
        # Save processed data
        output_data = []
        for pair in balanced_pairs:
            output_data.append({
                'code': pair.code,
                'comment': pair.comment,
                'language': pair.language,
                'quality_score': pair.quality_score,
                'augmented': pair.augmented
            })
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(output_data)} processed pairs to {output_file}")
        
        # Statistics
        python_count = len([p for p in balanced_pairs if p.language == 'python'])
        js_count = len([p for p in balanced_pairs if p.language == 'javascript'])
        augmented_count = len([p for p in balanced_pairs if p.augmented])
        avg_quality = sum(p.quality_score for p in balanced_pairs) / len(balanced_pairs)
        
        print(f"\n📊 Final Dataset Statistics:")
        print(f"   Total pairs: {len(balanced_pairs)}")
        print(f"   Python: {python_count} ({python_count/len(balanced_pairs)*100:.1f}%)")
        print(f"   JavaScript: {js_count} ({js_count/len(balanced_pairs)*100:.1f}%)")
        print(f"   Augmented: {augmented_count} ({augmented_count/len(balanced_pairs)*100:.1f}%)")
        print(f"   Average quality: {avg_quality:.3f}")

class EnhancedTokenizer:
    """Enhanced tokenizer for code and comments"""
    
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        self.start_token = 2
        self.end_token = 3
        self.pad_token = 0
        self.unk_token = 1
        
        # Special tokens
        self.special_tokens = {
            '<PAD>': self.pad_token,
            '<UNK>': self.unk_token,
            '<START>': self.start_token,
            '<END>': self.end_token
        }
        
        # Code-specific tokens
        self.code_tokens = [
            # Python keywords
            'def', 'class', 'if', 'else', 'elif', 'for', 'while', 'try', 'except',
            'import', 'from', 'return', 'yield', 'lambda', 'with', 'as', 'pass',
            'break', 'continue', 'global', 'nonlocal', 'assert', 'del', 'raise',
            'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is',
            
            # JavaScript keywords
            'function', 'const', 'let', 'var', 'async', 'await', 'export',
            'import', 'default', 'switch', 'case', 'break', 'continue',
            'true', 'false', 'null', 'undefined', 'typeof', 'instanceof',
            
            # Common symbols
            '(', ')', '[', ']', '{', '}', '.', ',', ':', ';', '=', '+', '-',
            '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', '&&', '||',
            '=>', '->', '...', '**', '//', '<<', '>>', '&', '|', '^', '~',
            
            # Common patterns
            'self', 'this', 'args', 'kwargs', 'params', 'options', 'config',
            'data', 'result', 'value', 'item', 'index', 'key', 'name',
            'type', 'length', 'size', 'count', 'total', 'max', 'min'
        ]
    
    def tokenize_text(self, text: str) -> List[str]:
        """Enhanced tokenization for code and comments"""
        # Handle code-specific patterns
        text = re.sub(r'([()[\]{}.,:;=+\-*/%<>!&|^~])', r' \1 ', text)
        text = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)', r' \1 ', text)
        text = re.sub(r'(\d+)', r' \1 ', text)
        text = re.sub(r'(["\'])', r' \1 ', text)
        
        # Split and clean
        tokens = text.split()
        tokens = [token.strip() for token in tokens if token.strip()]
        
        return tokens
    
    def build_vocab(self, texts: List[str], vocab_size: int = 10000):
        """Build vocabulary from texts"""
        print(f"🔤 Building vocabulary (target size: {vocab_size})...")
        
        # Count token frequencies
        token_counts = {}
        for text in texts:
            tokens = self.tokenize_text(text)
            for token in tokens:
                token_counts[token] = token_counts.get(token, 0) + 1
        
        # Start with special tokens
        self.vocab = self.special_tokens.copy()
        
        # Add code-specific tokens
        for token in self.code_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # Add most frequent tokens
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens:
            if len(self.vocab) >= vocab_size:
                break
            if token not in self.vocab and count >= 2:  # Minimum frequency
                self.vocab[token] = len(self.vocab)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        print(f"✅ Vocabulary built: {len(self.vocab)} tokens")
        return self.vocab
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize_text(text)
        return [self.vocab.get(token, self.unk_token) for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = [self.reverse_vocab.get(id, '<UNK>') for id in token_ids]
        # Remove special tokens
        tokens = [t for t in tokens if t not in ['<PAD>', '<START>', '<END>']]
        return ' '.join(tokens)

class EnhancedDataProcessor(DataProcessor):
    """Enhanced data processor with better tokenization"""
    
    def __init__(self):
        super().__init__()
        self.tokenizer = EnhancedTokenizer()
    
    def create_enhanced_tokenizer(self, data: List[Dict], vocab_size: int = 10000) -> EnhancedTokenizer:
        """Create enhanced tokenizer from data"""
        print("🔤 Creating enhanced tokenizer...")
        
        # Collect all text
        all_texts = []
        for item in data:
            all_texts.append(item['code'])
            all_texts.append(item['comment'])
        
        # Build vocabulary
        self.tokenizer.build_vocab(all_texts, vocab_size)
        
        return self.tokenizer
    
    def analyze_dataset(self, pairs: List[ProcessedPair]):
        """Analyze dataset statistics"""
        print("\n📊 Dataset Analysis:")
        
        # Language distribution
        lang_counts = {}
        for pair in pairs:
            lang_counts[pair.language] = lang_counts.get(pair.language, 0) + 1
        
        for lang, count in lang_counts.items():
            print(f"   {lang}: {count} pairs ({count/len(pairs)*100:.1f}%)")
        
        # Quality distribution
        quality_scores = [pair.quality_score for pair in pairs]
        print(f"   Quality: min={min(quality_scores):.3f}, max={max(quality_scores):.3f}, avg={sum(quality_scores)/len(quality_scores):.3f}")
        
        # Length statistics
        code_lengths = [len(pair.code.split()) for pair in pairs]
        comment_lengths = [len(pair.comment.split()) for pair in pairs]
        
        print(f"   Code length: min={min(code_lengths)}, max={max(code_lengths)}, avg={sum(code_lengths)/len(code_lengths):.1f}")
        print(f"   Comment length: min={min(comment_lengths)}, max={max(comment_lengths)}, avg={sum(comment_lengths)/len(comment_lengths):.1f}")
        
        # Augmentation statistics
        augmented_count = len([p for p in pairs if p.augmented])
        print(f"   Augmented: {augmented_count} pairs ({augmented_count/len(pairs)*100:.1f}%)")

def main():
    processor = EnhancedDataProcessor()
    
    # Process enhanced data
    input_files = [
        'data/enhanced_python_pairs.json',
        'data/enhanced_javascript_pairs.json', 
        'data/enhanced_all_pairs.json',
        'data/all_code_comments.json'  # Fallback to existing data
    ]
    
    # Check which files exist
    existing_files = [f for f in input_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No data files found. Please run data collection first.")
        return
    
    print(f"📁 Found data files: {existing_files}")
    
    processor.process_and_augment(
        input_files=existing_files,
        output_file='data/production_training_data.json',
        target_size=8000
    )

if __name__ == "__main__":
    main()