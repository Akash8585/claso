#!/usr/bin/env python3
"""
Extract model weights from checkpoint to avoid pickle issues
"""

import torch
import os
from dataclasses import dataclass

@dataclass
class FastTrainingConfig:
    """Fast training configuration for pickle compatibility"""
    model_type: str = 'comment'
    vocab_size: int = 8000
    max_seq_len: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-4
    num_epochs: int = 12
    warmup_steps: int = 500
    gradient_clip: float = 1.0
    use_mixed_precision: bool = False
    gradient_checkpointing: bool = False
    accumulation_steps: int = 2
    dropout: float = 0.05
    weight_decay: float = 0.005
    label_smoothing: float = 0.05
    validation_split: float = 0.1
    save_every_epochs: int = 3
    early_stopping_patience: int = 4
    data_path: str = 'data/enhanced_all_pairs.json'
    model_save_path: str = 'models/fast_enhanced'
    log_path: str = 'logs/fast_enhanced_training.log'

def extract_weights():
    """Extract just the model weights from checkpoint"""
    
    checkpoint_path = 'models/fast_comment/comment_best.pth'
    weights_path = 'models/fast_comment/comment_weights_only.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    try:
        # Load checkpoint with the config class available
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract just the model state dict
        if 'model_state_dict' in checkpoint:
            model_weights = checkpoint['model_state_dict']
        else:
            model_weights = checkpoint
        
        # Save just the weights
        torch.save(model_weights, weights_path)
        print(f"✅ Model weights extracted to: {weights_path}")
        
        # Print some info about the weights
        print(f"📊 Weight keys: {len(model_weights.keys())}")
        if 'src_embedding.weight' in model_weights:
            vocab_size, d_model = model_weights['src_embedding.weight'].shape
            print(f"🔤 Vocab size: {vocab_size}")
            print(f"🧠 Model dimension: {d_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to extract weights: {e}")
        return False

if __name__ == "__main__":
    extract_weights()