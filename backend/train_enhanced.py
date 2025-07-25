#!/usr/bin/env python3
"""
Enhanced training script for production-quality models
Supports 50M+ parameter models with advanced training techniques
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import pickle
import os
import time
import math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataclasses import dataclass

from model.production_transformer import EnhancedProductionTransformer
from data.data_processor import EnhancedDataProcessor

@dataclass
class TrainingConfig:
    """Enhanced training configuration"""
    # Model settings
    model_type: str = 'comment'  # 'comment' or 'commit'
    vocab_size: int = 10000
    max_seq_len: int = 1024
    
    # Training settings
    batch_size: int = 8  # Smaller for larger models
    learning_rate: float = 1e-4
    num_epochs: int = 30
    warmup_steps: int = 4000
    gradient_clip: float = 1.0
    
    # Advanced settings
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    accumulation_steps: int = 4  # Effective batch size = batch_size * accumulation_steps
    
    # Regularization
    dropout: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    
    # Validation and saving
    validation_split: float = 0.1
    save_every_epochs: int = 2
    early_stopping_patience: int = 5
    
    # Paths
    data_path: str = 'data/enhanced_all_pairs.json'
    model_save_path: str = 'models/enhanced'
    log_path: str = 'logs/enhanced_training.log'

class EnhancedCodeDataset(Dataset):
    """Enhanced dataset with better preprocessing"""
    
    def __init__(self, data: List[Dict], tokenizer, max_seq_len: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize code and comment
        code_tokens = self.tokenizer.encode(item['code'])
        comment_tokens = self.tokenizer.encode(item['comment'])
        
        # Truncate if too long
        if len(code_tokens) > self.max_seq_len - 2:
            code_tokens = code_tokens[:self.max_seq_len - 2]
        if len(comment_tokens) > self.max_seq_len - 2:
            comment_tokens = comment_tokens[:self.max_seq_len - 2]
        
        # Add special tokens
        src = [self.tokenizer.start_token] + code_tokens + [self.tokenizer.end_token]
        tgt_input = [self.tokenizer.start_token] + comment_tokens
        tgt_output = comment_tokens + [self.tokenizer.end_token]
        
        # Pad sequences
        src = self._pad_sequence(src, self.max_seq_len)
        tgt_input = self._pad_sequence(tgt_input, self.max_seq_len)
        tgt_output = self._pad_sequence(tgt_output, self.max_seq_len)
        
        return {
            'src': torch.tensor(src, dtype=torch.long),
            'tgt_input': torch.tensor(tgt_input, dtype=torch.long),
            'tgt_output': torch.tensor(tgt_output, dtype=torch.long),
            'quality_score': torch.tensor(item.get('quality_score', 1.0), dtype=torch.float)
        }
    
    def _pad_sequence(self, seq: List[int], max_len: int) -> List[int]:
        """Pad sequence to max length"""
        if len(seq) >= max_len:
            return seq[:max_len]
        return seq + [self.tokenizer.pad_token] * (max_len - len(seq))

class LabelSmoothingLoss(nn.Module):
    """Label smoothing loss for better generalization"""
    
    def __init__(self, vocab_size: int, smoothing: float = 0.1, ignore_index: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred: (batch_size, seq_len, vocab_size)
        target: (batch_size, seq_len)
        """
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)
        
        # Create smoothed labels
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.smoothing / (self.vocab_size - 1))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Mask padding tokens
        mask = (target != self.ignore_index)
        true_dist = true_dist * mask.unsqueeze(1).float()
        
        # Calculate loss
        log_pred = torch.log_softmax(pred, dim=1)
        loss = -torch.sum(true_dist * log_pred, dim=1)
        
        return loss[mask].mean()

class WarmupScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, d_model: int, warmup_steps: int = 4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0
        
    def step(self):
        self.step_num += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def _get_lr(self):
        return (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )

class EnhancedTrainer:
    """Enhanced trainer with advanced techniques"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        os.makedirs(os.path.dirname(config.log_path), exist_ok=True)
        self.log_file = open(config.log_path, 'w')
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'perplexity': []
        }
        
        print(f"🚀 Enhanced Trainer initialized")
        print(f"   🎯 Model type: {config.model_type}")
        print(f"   💻 Device: {self.device}")
        print(f"   🔥 Mixed precision: {config.use_mixed_precision}")
        print(f"   💾 Gradient checkpointing: {config.gradient_checkpointing}")
        
    def log(self, message: str):
        """Log message to file and console"""
        print(message)
        self.log_file.write(f"{message}\n")
        self.log_file.flush()
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, object]:
        """Load and prepare data"""
        self.log("📊 Loading enhanced training data...")
        
        # Check for multiple data sources in priority order
        data_sources = [
            self.config.data_path,  # Primary: production_training_data.json
            'data/enhanced_all_pairs.json',  # Advanced collector output
            'data/enhanced_python_pairs.json',  # Python-specific
            'data/enhanced_javascript_pairs.json',  # JS-specific
            'data/all_code_comments.json'  # Fallback
        ]
        
        all_data = []
        for data_path in data_sources:
            if os.path.exists(data_path):
                self.log(f"📁 Loading data from: {data_path}")
                try:
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                        
                    # Handle different data formats
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], list):  # Old format [code, comment]
                            for item in data:
                                if len(item) >= 2:
                                    all_data.append({
                                        'code': item[0],
                                        'comment': item[1],
                                        'language': 'python' if 'python' in data_path else 'javascript',
                                        'quality_score': 0.7,
                                        'comment_type': 'docstring'
                                    })
                        else:  # New format with metadata
                            all_data.extend(data)
                    
                    self.log(f"   ✅ Loaded {len(data)} samples from {data_path}")
                    
                    # If we have enough data, break
                    if len(all_data) >= 1000:
                        break
                        
                except Exception as e:
                    self.log(f"   ⚠️  Error loading {data_path}: {e}")
                    continue
        
        if not all_data:
            self.log("❌ No training data found!")
            self.log("   Available data sources checked:")
            for source in data_sources:
                exists = "✅" if os.path.exists(source) else "❌"
                self.log(f"   {exists} {source}")
            raise FileNotFoundError("No training data available")
        
        self.log(f"   📈 Total samples: {len(all_data)}")
        
        # Filter by model type if needed
        if self.config.model_type == 'comment':
            filtered_data = [item for item in all_data if item.get('comment_type') in ['docstring', 'jsdoc']]
        else:
            filtered_data = all_data  # Use all data for commit messages
        
        self.log(f"   🎯 Filtered samples: {len(filtered_data)}")
        
        # Create or load tokenizer
        processor = EnhancedDataProcessor()
        tokenizer = processor.create_enhanced_tokenizer(filtered_data, vocab_size=self.config.vocab_size)
        
        # Split data
        val_size = int(len(filtered_data) * self.config.validation_split)
        train_data = filtered_data[val_size:]
        val_data = filtered_data[:val_size]
        
        self.log(f"   🚂 Training samples: {len(train_data)}")
        self.log(f"   ✅ Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = EnhancedCodeDataset(train_data, tokenizer, self.config.max_seq_len)
        val_dataset = EnhancedCodeDataset(val_data, tokenizer, self.config.max_seq_len)
        
        # Create data loaders - fix for Apple Silicon
        pin_memory = torch.cuda.is_available()  # Only pin memory for CUDA
        num_workers = 0 if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 2
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, tokenizer
    
    def create_model(self, tokenizer):
        """Create enhanced model"""
        self.log(f"🧠 Creating enhanced {self.config.model_type} model...")
        
        # Use fast model for better speed/performance balance
        from model.production_transformer import FastEnhancedTransformer
        
        model = FastEnhancedTransformer(
            vocab_size=len(tokenizer.vocab),
            model_type=self.config.model_type,
            max_seq_len=self.config.max_seq_len,
            dropout=self.config.dropout
        )
        
        if self.config.gradient_checkpointing:
            model.enable_gradient_checkpointing()
            self.log("   ✅ Gradient checkpointing enabled")
        
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"   📊 Total parameters: {total_params:,}")
        self.log(f"   🎯 Trainable parameters: {trainable_params:,}")
        self.log(f"   💾 Model size: ~{total_params * 4 / 1024**2:.1f} MB")
        
        return model
    
    def setup_training(self, model) -> Tuple[optim.Optimizer, object, object]:
        """Setup optimizer, scheduler, and loss function"""
        # Optimizer with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        scheduler = WarmupScheduler(
            optimizer,
            d_model=model.d_model,
            warmup_steps=self.config.warmup_steps
        )
        
        # Loss function with label smoothing
        criterion = LabelSmoothingLoss(
            vocab_size=model.vocab_size,
            smoothing=self.config.label_smoothing,
            ignore_index=0  # Padding token
        )
        
        # Mixed precision scaler - fix for Apple Silicon
        if self.config.use_mixed_precision:
            if torch.cuda.is_available():
                scaler = torch.cuda.amp.GradScaler()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                scaler = None  # MPS doesn't support GradScaler yet
                self.config.use_mixed_precision = False  # Disable for MPS
                self.log("   ⚠️  Mixed precision disabled for Apple Silicon")
            else:
                scaler = None
                self.config.use_mixed_precision = False
        else:
            scaler = None
        
        self.log("⚙️  Training setup completed:")
        self.log(f"   📈 Learning rate: {self.config.learning_rate}")
        self.log(f"   🔄 Warmup steps: {self.config.warmup_steps}")
        self.log(f"   ✂️  Gradient clipping: {self.config.gradient_clip}")
        self.log(f"   🎯 Label smoothing: {self.config.label_smoothing}")
        
        return optimizer, scheduler, criterion, scaler
    
    def train_epoch(self, model, train_loader, optimizer, scheduler, criterion, scaler, epoch):
        """Train one epoch"""
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision and scaler and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    outputs = model(src, tgt_input)
                    loss = criterion(outputs, tgt_output)
                    loss = loss / self.config.accumulation_steps
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1
            else:
                outputs = model(src, tgt_input)
                loss = criterion(outputs, tgt_output)
                loss = loss / self.config.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1
            
            total_loss += loss.item() * self.config.accumulation_steps
            
            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * self.config.accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}',
                'step': self.global_step
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, model, val_loader, criterion):
        """Validate model"""
        model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                outputs = model(src, tgt_input)
                loss = criterion(outputs, tgt_output)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, model, tokenizer, epoch, loss, is_best=False):
        """Save model checkpoint"""
        os.makedirs(self.config.model_save_path, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'config': self.config,
            'global_step': self.global_step,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config.model_save_path, 
            f'{self.config.model_type}_epoch_{epoch+1}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(
            self.config.model_save_path,
            f'{self.config.model_type}_tokenizer.pkl'
        )
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config.model_save_path,
                f'{self.config.model_type}_best.pth'
            )
            torch.save(checkpoint, best_path)
            self.log(f"💾 Best model saved: {best_path}")
        
        self.log(f"💾 Checkpoint saved: {checkpoint_path}")
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.training_history['train_loss']:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        epochs = range(1, len(self.training_history['train_loss']) + 1)
        ax1.plot(epochs, self.training_history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.training_history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Learning rate
        ax2.plot(epochs, self.training_history['learning_rate'], 'g-')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # Perplexity
        ax3.plot(epochs, self.training_history['perplexity'], 'purple')
        ax3.set_title('Validation Perplexity')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Perplexity')
        ax3.grid(True)
        
        # Loss comparison
        ax4.semilogy(epochs, self.training_history['train_loss'], 'b-', label='Training')
        ax4.semilogy(epochs, self.training_history['val_loss'], 'r-', label='Validation')
        ax4.set_title('Loss (Log Scale)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss (log)')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.config.model_save_path, f'{self.config.model_type}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.log(f"📊 Training history plot saved: {plot_path}")
    
    def train(self):
        """Main training loop"""
        self.log("🚀 Starting enhanced training...")
        start_time = time.time()
        
        # Load data
        train_loader, val_loader, tokenizer = self.load_data()
        
        # Create model
        model = self.create_model(tokenizer)
        
        # Setup training
        optimizer, scheduler, criterion, scaler = self.setup_training(model)
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(
                model, train_loader, optimizer, scheduler, criterion, scaler, epoch
            )
            
            # Validate
            val_loss, perplexity = self.validate(model, val_loader, criterion)
            
            # Update history
            current_lr = optimizer.param_groups[0]['lr']
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['perplexity'].append(perplexity)
            
            # Log progress
            epoch_time = time.time() - epoch_start
            self.log(f"\n📊 Epoch {epoch+1}/{self.config.num_epochs} Summary:")
            self.log(f"   🚂 Train Loss: {train_loss:.4f}")
            self.log(f"   ✅ Val Loss: {val_loss:.4f}")
            self.log(f"   🎯 Perplexity: {perplexity:.2f}")
            self.log(f"   📈 Learning Rate: {current_lr:.2e}")
            self.log(f"   ⏱️  Time: {epoch_time:.1f}s")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % self.config.save_every_epochs == 0 or is_best:
                self.save_checkpoint(model, tokenizer, epoch, val_loss, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                self.log(f"🛑 Early stopping triggered after {epoch+1} epochs")
                break
            
            # Plot training history
            if (epoch + 1) % 5 == 0:
                self.plot_training_history()
        
        # Final save and summary
        self.save_checkpoint(model, tokenizer, epoch, val_loss, False)
        self.plot_training_history()
        
        total_time = time.time() - start_time
        self.log(f"\n🎉 Training completed!")
        self.log(f"   ⏱️  Total time: {total_time/3600:.1f} hours")
        self.log(f"   🏆 Best validation loss: {self.best_val_loss:.4f}")
        self.log(f"   📊 Total steps: {self.global_step}")
        
        self.log_file.close()

def main():
    """Main training function"""
    # Comment model configuration
    comment_config = TrainingConfig(
        model_type='comment',
        batch_size=6,  # Smaller batch for 50M model
        learning_rate=5e-5,
        num_epochs=25,
        data_path='data/enhanced_all_pairs.json',
        model_save_path='models/enhanced_comment',
        log_path='logs/enhanced_comment_training.log'
    )
    
    # Train comment model
    print("🚀 Training Enhanced Comment Model (50M parameters)")
    trainer = EnhancedTrainer(comment_config)
    trainer.train()
    
    # Commit model configuration
    commit_config = TrainingConfig(
        model_type='commit',
        batch_size=4,  # Even smaller batch for 80M model
        learning_rate=3e-5,
        num_epochs=25,
        data_path='data/enhanced_all_pairs.json',
        model_save_path='models/enhanced_commit',
        log_path='logs/enhanced_commit_training.log'
    )
    
    # Train commit model
    print("\n🚀 Training Enhanced Commit Model (80M parameters)")
    trainer = EnhancedTrainer(commit_config)
    trainer.train()
    
    print("\n🎉 All enhanced models trained successfully!")

if __name__ == "__main__":
    main()