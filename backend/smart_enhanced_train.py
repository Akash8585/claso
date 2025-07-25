#!/usr/bin/env python3
"""
Smart Enhanced Training Launcher
Automatically detects available data and starts appropriate training
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

def setup_logging():
    """Setup logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_data_availability(logger):
    """Check what training data is available"""
    logger.info("🔍 Checking available training data...")
    
    data_sources = [
        ('data/enhanced_all_pairs.json', 'Advanced collected data'),
        ('data/enhanced_python_pairs.json', 'Python-specific data'),
        ('data/enhanced_javascript_pairs.json', 'JavaScript-specific data'),
        ('data/production_training_data.json', 'Processed training data'),
        ('data/all_code_comments.json', 'Original training data')
    ]
    
    available_data = []
    total_samples = 0
    
    for file_path, description in data_sources:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    samples = len(data)
                    total_samples += samples
                    available_data.append((file_path, description, samples))
                    logger.info(f"   ✅ {description}: {samples} samples")
            except Exception as e:
                logger.warning(f"   ⚠️  Error reading {file_path}: {e}")
        else:
            logger.info(f"   ❌ {description}: Not found")
    
    logger.info(f"📊 Total available samples: {total_samples}")
    return available_data, total_samples

def wait_for_data_collection(logger):
    """Check if data collection is in progress and optionally wait"""
    logger.info("⏳ Checking if data collection is in progress...")
    
    # Check if advanced data collector output exists and is growing
    target_files = [
        'data/enhanced_python_pairs.json',
        'data/enhanced_javascript_pairs.json',
        'data/enhanced_all_pairs.json'
    ]
    
    # Check if any target files exist and are recent
    recent_files = []
    for file_path in target_files:
        if os.path.exists(file_path):
            file_age = time.time() - os.path.getmtime(file_path)
            if file_age < 300:  # Modified in last 5 minutes
                recent_files.append(file_path)
    
    if recent_files:
        logger.info(f"📊 Data collection appears to be in progress: {recent_files}")
        logger.info("   But we have sufficient data to start training now!")
        logger.info("   Proceeding with available data...")
    else:
        logger.info("✅ No active data collection detected")

def recommend_training_approach(available_data, total_samples, logger):
    """Recommend best training approach based on available data"""
    logger.info("🎯 Analyzing training approach...")
    
    if total_samples >= 3000:
        logger.info("🚀 Excellent! Sufficient data for full enhanced training")
        return "full_enhanced"
    elif total_samples >= 1500:
        logger.info("✅ Good data available - enhanced training with smaller models")
        return "enhanced_medium"
    elif total_samples >= 500:
        logger.info("⚠️  Limited data - will use smaller models and more epochs")
        return "enhanced_small"
    else:
        logger.info("❌ Insufficient data for enhanced training")
        logger.info("   Please run data collection first:")
        logger.info("   python tools/advanced_data_collector.py")
        return "insufficient"

def create_adaptive_config(approach, total_samples):
    """Create training config based on data availability"""
    base_config = {
        "vocab_size": min(8000, max(5000, total_samples // 3)),  # Reduced vocab
        "max_seq_len": 512,  # Reduced context for speed
        "gradient_clip": 1.0,
        "use_mixed_precision": False,  # Disabled for Apple Silicon compatibility
        "gradient_checkpointing": False,  # Disabled for speed
        "accumulation_steps": 2,  # Reduced accumulation
        "dropout": 0.05,  # Reduced dropout
        "weight_decay": 0.005,  # Reduced weight decay
        "label_smoothing": 0.05,  # Reduced smoothing
        "validation_split": 0.1,
        "save_every_epochs": 3,
        "early_stopping_patience": 4
    }
    
    if approach == "full_enhanced":
        # Fast enhanced models optimized for 1-1.5 hours
        configs = {
            "comment": {
                **base_config,
                "model_type": "comment",
                "batch_size": 20,  # Larger batch for speed
                "learning_rate": 2e-4,  # Higher LR for faster convergence
                "num_epochs": 10,  # Reduced epochs
                "warmup_steps": 500  # Reduced warmup
            },
            "commit": {
                **base_config,
                "model_type": "commit", 
                "batch_size": 16,  # Larger batch
                "learning_rate": 1.5e-4,  # Higher LR
                "num_epochs": 8,  # Reduced epochs
                "warmup_steps": 400  # Reduced warmup
            }
        }
    elif approach == "enhanced_medium":
        # Medium models optimized for speed
        configs = {
            "comment": {
                **base_config,
                "model_type": "comment",
                "batch_size": 16,  # Increased batch
                "learning_rate": 1.5e-4,  # Higher LR
                "num_epochs": 12,  # Reduced epochs
                "warmup_steps": 800  # Reduced warmup
            },
            "commit": {
                **base_config,
                "model_type": "commit",
                "batch_size": 12,  # Increased batch
                "learning_rate": 1e-4,  # Higher LR
                "num_epochs": 10,  # Reduced epochs
                "warmup_steps": 600  # Reduced warmup
            }
        }
    else:  # enhanced_small
        # Smaller models, optimized training
        configs = {
            "comment": {
                **base_config,
                "model_type": "comment",
                "batch_size": 24,  # Large batch for speed
                "learning_rate": 3e-4,  # High LR
                "num_epochs": 15,  # Moderate epochs
                "warmup_steps": 300  # Minimal warmup
            },
            "commit": {
                **base_config,
                "model_type": "commit",
                "batch_size": 16,  # Large batch
                "learning_rate": 2e-4,  # High LR
                "num_epochs": 12,  # Moderate epochs
                "warmup_steps": 250  # Minimal warmup
            }
        }
    
    return configs

def start_training(approach, configs, logger):
    """Start the appropriate training"""
    logger.info(f"🚀 Starting {approach} training...")
    
    # Import training modules
    try:
        from train_enhanced import EnhancedTrainer, TrainingConfig
    except ImportError as e:
        logger.error(f"❌ Failed to import training modules: {e}")
        return False
    
    # Train comment model
    logger.info("📝 Training Comment Model...")
    comment_config = TrainingConfig(**configs["comment"])
    comment_config.model_save_path = 'models/enhanced_comment'
    comment_config.log_path = 'logs/enhanced_comment_training.log'
    
    try:
        comment_trainer = EnhancedTrainer(comment_config)
        comment_trainer.train()
        logger.info("✅ Comment model training completed")
    except Exception as e:
        logger.error(f"❌ Comment model training failed: {e}")
        return False
    
    # Train commit model
    logger.info("💬 Training Commit Model...")
    commit_config = TrainingConfig(**configs["commit"])
    commit_config.model_save_path = 'models/enhanced_commit'
    commit_config.log_path = 'logs/enhanced_commit_training.log'
    
    try:
        commit_trainer = EnhancedTrainer(commit_config)
        commit_trainer.train()
        logger.info("✅ Commit model training completed")
    except Exception as e:
        logger.error(f"❌ Commit model training failed: {e}")
        return False
    
    return True

def main():
    """Main smart training function"""
    logger = setup_logging()
    
    logger.info("🧠 Smart Enhanced Training Launcher")
    logger.info("=" * 50)
    
    # Check for --no-wait flag
    skip_wait = len(sys.argv) > 1 and '--no-wait' in sys.argv
    
    if skip_wait:
        logger.info("⚡ Skipping data collection check (--no-wait flag)")
    else:
        # Wait for data collection if in progress
        wait_for_data_collection(logger)
    
    # Check available data
    available_data, total_samples = check_data_availability(logger)
    
    if total_samples == 0:
        logger.error("❌ No training data found!")
        logger.error("   Please run data collection:")
        logger.error("   python tools/advanced_data_collector.py")
        return False
    
    # Recommend approach
    approach = recommend_training_approach(available_data, total_samples, logger)
    
    if approach == "insufficient":
        return False
    
    # Create adaptive config
    configs = create_adaptive_config(approach, total_samples)
    
    # Show training plan
    logger.info("📋 Training Plan:")
    logger.info(f"   🎯 Approach: {approach}")
    logger.info(f"   📊 Data samples: {total_samples}")
    logger.info(f"   📝 Comment model: {configs['comment']['num_epochs']} epochs")
    logger.info(f"   💬 Commit model: {configs['commit']['num_epochs']} epochs")
    
    # Estimate time
    if approach == "full_enhanced":
        estimated_time = "2-3 hours"
    elif approach == "enhanced_medium":
        estimated_time = "1.5-2 hours"
    else:
        estimated_time = "1-1.5 hours"
    
    logger.info(f"   ⏱️  Estimated time: {estimated_time}")
    
    # Start training
    success = start_training(approach, configs, logger)
    
    if success:
        logger.info("🎉 Enhanced training completed successfully!")
        logger.info("📁 Check models/enhanced_*/ for trained models")
        logger.info("📊 Check logs/ for training logs")
        logger.info("📈 Check plots/ for training visualizations")
    else:
        logger.error("❌ Training failed")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)