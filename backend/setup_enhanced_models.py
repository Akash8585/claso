#!/usr/bin/env python3
"""
Enhanced Model Setup Script
Complete pipeline for setting up production-quality ML models
Handles data processing, model creation, and training with full logging
"""

import os
import sys
import json
import time
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"enhanced_setup_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🚀 Enhanced Model Setup Started")
    logger.info(f"📝 Log file: {log_file}")
    logger.info("=" * 60)
    
    return logger

def check_system_requirements(logger: logging.Logger) -> bool:
    """Check system requirements and log results"""
    logger.info("🔍 Checking System Requirements...")
    
    # Check Python version
    python_version = sys.version_info
    logger.info(f"   🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        logger.error("❌ Python 3.8+ required")
        return False
    
    # Check required packages
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'matplotlib': 'Matplotlib',
        'tqdm': 'Progress bars',
        'requests': 'HTTP requests'
    }
    
    missing_packages = []
    for package, description in required_packages.items():
        try:
            __import__(package)
            logger.info(f"   ✅ {description}: Available")
        except ImportError:
            logger.error(f"   ❌ {description}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"❌ Missing packages: {missing_packages}")
        logger.error("   Install with: pip install " + " ".join(missing_packages))
        return False
    
    # Check GPU
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"   🚀 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("   🚀 Apple Silicon GPU: Available")
        else:
            logger.warning("   ⚠️  No GPU detected - will use CPU (slower)")
    except Exception as e:
        logger.warning(f"   ⚠️  GPU check failed: {e}")
    
    logger.info("✅ System requirements check completed")
    return True

def setup_directories(logger: logging.Logger) -> bool:
    """Create all necessary directories"""
    logger.info("📁 Setting up directories...")
    
    directories = [
        "logs",
        "models",
        "models/enhanced_comment", 
        "models/enhanced_commit",
        "data",
        "plots",
        "checkpoints",
        "backups"
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"   ✅ Created: {directory}")
        except Exception as e:
            logger.error(f"   ❌ Failed to create {directory}: {e}")
            return False
    
    logger.info("✅ Directory setup completed")
    return True

def check_and_process_data(logger: logging.Logger) -> bool:
    """Check existing data and process it for enhanced training"""
    logger.info("📊 Checking and processing training data...")
    
    # Check for existing data files
    data_files = [
        "data/all_code_comments.json",
        "data/python_code_comments.json", 
        "data/javascript_code_comments.json"
    ]
    
    existing_files = []
    total_samples = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    samples = len(data)
                    total_samples += samples
                    existing_files.append(file_path)
                    logger.info(f"   ✅ {file_path}: {samples} samples")
            except Exception as e:
                logger.error(f"   ❌ Error reading {file_path}: {e}")
        else:
            logger.warning(f"   ⚠️  {file_path}: Not found")
    
    if not existing_files:
        logger.error("❌ No training data found!")
        logger.error("   Please run data collection first:")
        logger.error("   python tools/advanced_data_collector.py")
        return False
    
    logger.info(f"📊 Total available samples: {total_samples}")
    
    # Process data using enhanced processor
    logger.info("🔄 Processing data with enhanced processor...")
    
    try:
        # Import here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from data.data_processor import EnhancedDataProcessor
        
        processor = EnhancedDataProcessor()
        
        # Convert existing data to enhanced format
        enhanced_data = []
        for file_path in existing_files:
            logger.info(f"   📄 Processing {file_path}...")
            
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            
            # Convert format based on file structure
            if isinstance(raw_data[0], list):  # Old format [code, comment]
                for item in raw_data:
                    if len(item) >= 2:
                        enhanced_data.append({
                            'code': item[0],
                            'comment': item[1],
                            'language': 'python' if 'python' in file_path else 'javascript',
                            'quality_score': 0.7,  # Default score
                            'comment_type': 'docstring' if 'python' in file_path else 'jsdoc',
                            'repo': 'existing_data',
                            'file_path': 'processed'
                        })
            else:  # New format with metadata
                enhanced_data.extend(raw_data)
        
        logger.info(f"   📊 Converted {len(enhanced_data)} samples to enhanced format")
        
        # Save enhanced data
        enhanced_file = "data/enhanced_all_pairs.json"
        with open(enhanced_file, 'w') as f:
            json.dump(enhanced_data, f, indent=2)
        
        logger.info(f"   💾 Saved enhanced data: {enhanced_file}")
        
        # Process and augment data
        logger.info("🔄 Running data augmentation...")
        
        processor.process_and_augment(
            input_files=[enhanced_file],
            output_file='data/production_training_data.json',
            target_size=min(5000, len(enhanced_data) * 3)  # 3x augmentation
        )
        
        # Verify processed data
        if os.path.exists('data/production_training_data.json'):
            with open('data/production_training_data.json', 'r') as f:
                processed_data = json.load(f)
            logger.info(f"✅ Final training data: {len(processed_data)} samples")
            return True
        else:
            logger.error("❌ Failed to create processed training data")
            return False
            
    except Exception as e:
        logger.error(f"❌ Data processing failed: {e}")
        logger.error(f"   Error details: {str(e)}")
        return False

def create_training_configs(logger: logging.Logger) -> bool:
    """Create training configuration files"""
    logger.info("⚙️  Creating training configurations...")
    
    # Comment model config
    comment_config = {
        "model_type": "comment",
        "vocab_size": 10000,
        "max_seq_len": 1024,
        "batch_size": 6,
        "learning_rate": 5e-5,
        "num_epochs": 20,
        "warmup_steps": 2000,
        "gradient_clip": 1.0,
        "use_mixed_precision": True,
        "gradient_checkpointing": True,
        "accumulation_steps": 4,
        "dropout": 0.1,
        "weight_decay": 0.01,
        "label_smoothing": 0.1,
        "validation_split": 0.1,
        "save_every_epochs": 2,
        "early_stopping_patience": 5
    }
    
    # Commit model config  
    commit_config = comment_config.copy()
    commit_config.update({
        "model_type": "commit",
        "batch_size": 4,  # Smaller for larger model
        "learning_rate": 3e-5,
        "num_epochs": 15
    })
    
    # Save configs
    try:
        with open('models/enhanced_comment/config.json', 'w') as f:
            json.dump(comment_config, f, indent=2)
        logger.info("   ✅ Comment model config saved")
        
        with open('models/enhanced_commit/config.json', 'w') as f:
            json.dump(commit_config, f, indent=2)
        logger.info("   ✅ Commit model config saved")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to save configs: {e}")
        return False

def estimate_training_time(logger: logging.Logger):
    """Estimate and log training times"""
    logger.info("⏱️  Estimating training times...")
    
    try:
        import torch
        
        # Detect hardware
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            if "RTX" in gpu_name or "GTX" in gpu_name:
                device_type = "NVIDIA GPU"
                comment_time = "30-45 minutes"
                commit_time = "45-75 minutes"
            else:
                device_type = "GPU"
                comment_time = "45-60 minutes" 
                commit_time = "60-90 minutes"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_type = "Apple Silicon"
            comment_time = "45-60 minutes"
            commit_time = "60-90 minutes"
        else:
            device_type = "CPU"
            comment_time = "3-4 hours"
            commit_time = "4-6 hours"
        
        logger.info(f"   🖥️  Hardware: {device_type}")
        logger.info(f"   📊 Comment Model (50M params): {comment_time}")
        logger.info(f"   📊 Commit Model (80M params): {commit_time}")
        logger.info(f"   📊 Total pipeline: ~2-3 hours on {device_type}")
        
    except Exception as e:
        logger.warning(f"   ⚠️  Could not estimate times: {e}")

def create_training_scripts(logger: logging.Logger) -> bool:
    """Create convenient training launcher scripts"""
    logger.info("📝 Creating training launcher scripts...")
    
    # Quick training script
    quick_train_script = '''#!/usr/bin/env python3
"""
Quick Enhanced Training Launcher
"""
import os
import sys
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("🚀 Quick Enhanced Training Started")
    
    # Check data
    if not os.path.exists('data/production_training_data.json'):
        logger.error("❌ Training data not found!")
        logger.error("   Run: python setup_enhanced_models.py")
        return
    
    # Start training
    logger.info("🧠 Starting enhanced model training...")
    exit_code = os.system('python train_enhanced.py')
    
    if exit_code == 0:
        logger.info("🎉 Training completed successfully!")
    else:
        logger.error("❌ Training failed!")

if __name__ == "__main__":
    main()
'''
    
    # Step-by-step training script
    step_train_script = '''#!/usr/bin/env python3
"""
Step-by-step Enhanced Training
Train models one at a time with user confirmation
"""
import os
import sys
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def main():
    logger = setup_logging()
    logger.info("🚀 Step-by-step Enhanced Training")
    
    # Check data
    if not os.path.exists('data/production_training_data.json'):
        logger.error("❌ Training data not found!")
        logger.error("   Run: python setup_enhanced_models.py")
        return
    
    # Train comment model
    logger.info("\\n📝 Ready to train Comment Model (50M parameters)")
    logger.info("   Estimated time: 45-60 minutes")
    input("Press Enter to start comment model training...")
    
    os.system('python -c "from train_enhanced import main, TrainingConfig; config = TrainingConfig(model_type=\\'comment\\'); from train_enhanced import EnhancedTrainer; trainer = EnhancedTrainer(config); trainer.train()"')
    
    # Train commit model
    logger.info("\\n💬 Ready to train Commit Model (80M parameters)")
    logger.info("   Estimated time: 60-90 minutes")
    input("Press Enter to start commit model training...")
    
    os.system('python -c "from train_enhanced import main, TrainingConfig; config = TrainingConfig(model_type=\\'commit\\'); from train_enhanced import EnhancedTrainer; trainer = EnhancedTrainer(config); trainer.train()"')
    
    logger.info("🎉 All models trained!")

if __name__ == "__main__":
    main()
'''
    
    try:
        # Save scripts
        with open('quick_enhanced_train.py', 'w') as f:
            f.write(quick_train_script)
        os.chmod('quick_enhanced_train.py', 0o755)
        logger.info("   ✅ Created: quick_enhanced_train.py")
        
        with open('step_enhanced_train.py', 'w') as f:
            f.write(step_train_script)
        os.chmod('step_enhanced_train.py', 0o755)
        logger.info("   ✅ Created: step_enhanced_train.py")
        
        return True
    except Exception as e:
        logger.error(f"❌ Failed to create scripts: {e}")
        return False

def backup_existing_models(logger: logging.Logger):
    """Backup existing models before training new ones"""
    logger.info("💾 Backing up existing models...")
    
    model_dirs = ["models/comment", "models/commit"]
    backup_dir = f"backups/models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    backed_up = False
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            try:
                os.makedirs(backup_dir, exist_ok=True)
                backup_path = os.path.join(backup_dir, os.path.basename(model_dir))
                shutil.copytree(model_dir, backup_path)
                logger.info(f"   ✅ Backed up: {model_dir} -> {backup_path}")
                backed_up = True
            except Exception as e:
                logger.warning(f"   ⚠️  Failed to backup {model_dir}: {e}")
    
    if not backed_up:
        logger.info("   ℹ️  No existing models to backup")

def create_monitoring_script(logger: logging.Logger):
    """Create training monitoring script"""
    logger.info("📊 Creating training monitor...")
    
    monitor_script = '''#!/usr/bin/env python3
"""
Enhanced Training Monitor
Monitor training progress and logs
"""
import os
import time
import json
from pathlib import Path

def monitor_training():
    print("📊 Enhanced Training Monitor")
    print("=" * 40)
    
    log_files = [
        "logs/enhanced_comment_training.log",
        "logs/enhanced_commit_training.log"
    ]
    
    model_dirs = [
        "models/enhanced_comment",
        "models/enhanced_commit"
    ]
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print("📊 Enhanced Training Monitor")
        print("=" * 40)
        print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Check log files
        for log_file in log_files:
            if os.path.exists(log_file):
                model_type = "Comment" if "comment" in log_file else "Commit"
                print(f"📝 {model_type} Model:")
                
                # Get last few lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-3:]:
                        print(f"   {line.strip()}")
                print()
        
        # Check model files
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                model_type = "Comment" if "comment" in model_dir else "Commit"
                files = list(Path(model_dir).glob("*.pth"))
                print(f"💾 {model_type} Checkpoints: {len(files)}")
        
        print("\\nPress Ctrl+C to exit")
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\\n👋 Monitor stopped")
'''
    
    try:
        with open('monitor_training.py', 'w') as f:
            f.write(monitor_script)
        os.chmod('monitor_training.py', 0o755)
        logger.info("   ✅ Created: monitor_training.py")
    except Exception as e:
        logger.warning(f"   ⚠️  Failed to create monitor: {e}")

def main():
    """Main setup function"""
    # Setup logging first
    logger = setup_logging()
    
    start_time = time.time()
    
    try:
        # Step 1: Check system requirements
        if not check_system_requirements(logger):
            logger.error("❌ System requirements not met")
            return False
        
        # Step 2: Setup directories
        if not setup_directories(logger):
            logger.error("❌ Directory setup failed")
            return False
        
        # Step 3: Backup existing models
        backup_existing_models(logger)
        
        # Step 4: Check and process data
        if not check_and_process_data(logger):
            logger.error("❌ Data processing failed")
            return False
        
        # Step 5: Create training configs
        if not create_training_configs(logger):
            logger.error("❌ Config creation failed")
            return False
        
        # Step 6: Estimate training time
        estimate_training_time(logger)
        
        # Step 7: Create training scripts
        if not create_training_scripts(logger):
            logger.warning("⚠️  Script creation failed (non-critical)")
        
        # Step 8: Create monitoring tools
        create_monitoring_script(logger)
        
        # Success summary
        setup_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info("🎉 Enhanced Model Setup Completed Successfully!")
        logger.info(f"⏱️  Setup time: {setup_time:.1f} seconds")
        logger.info("")
        logger.info("📋 Next Steps:")
        logger.info("1. Start training:")
        logger.info("   python train_enhanced.py           # Full training")
        logger.info("   python quick_enhanced_train.py     # Quick launcher")
        logger.info("   python step_enhanced_train.py      # Step-by-step")
        logger.info("")
        logger.info("2. Monitor progress:")
        logger.info("   python monitor_training.py         # Live monitor")
        logger.info("   tail -f logs/enhanced_*_training.log  # Log files")
        logger.info("")
        logger.info("3. Check results:")
        logger.info("   ls models/enhanced_*/               # Model files")
        logger.info("   ls plots/                          # Training plots")
        logger.info("")
        logger.info("🚀 Ready for production-quality ML training!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed with error: {e}")
        logger.error(f"   Error details: {str(e)}")
        return False
    
    finally:
        logger.info("=" * 60)
        logger.info("📝 Setup log saved for reference")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)