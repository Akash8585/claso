#!/usr/bin/env python3
"""
Extract model weights from training checkpoint
Fixes the FastTrainingConfig loading issue
"""

import torch
import pickle
import os

def extract_model_weights():
    """Extract just the model weights from training checkpoint"""
    
    # Comment model
    comment_checkpoint_path = 'models/fast_comment/comment_best.pth'
    comment_weights_path = 'models/fast_comment/comment_weights.pth'
    
    if os.path.exists(comment_checkpoint_path):
        print("📝 Extracting comment model weights...")
        try:
            # Load with weights_only=False to get the full checkpoint
            checkpoint = torch.load(comment_checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract just the model state dict
            if 'model_state_dict' in checkpoint:
                model_weights = checkpoint['model_state_dict']
            else:
                model_weights = checkpoint
            
            # Save just the weights
            torch.save(model_weights, comment_weights_path)
            print(f"✅ Comment model weights saved to: {comment_weights_path}")
            
        except Exception as e:
            print(f"❌ Error extracting comment model: {e}")
    
    # Commit model (if exists)
    commit_checkpoint_path = 'models/fast_commit/commit_best.pth'
    commit_weights_path = 'models/fast_commit/commit_weights.pth'
    
    if os.path.exists(commit_checkpoint_path):
        print("💬 Extracting commit model weights...")
        try:
            checkpoint = torch.load(commit_checkpoint_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in checkpoint:
                model_weights = checkpoint['model_state_dict']
            else:
                model_weights = checkpoint
            
            torch.save(model_weights, commit_weights_path)
            print(f"✅ Commit model weights saved to: {commit_weights_path}")
            
        except Exception as e:
            print(f"❌ Error extracting commit model: {e}")

if __name__ == "__main__":
    extract_model_weights()