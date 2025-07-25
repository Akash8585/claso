#!/usr/bin/env python3
"""
Setup script for enhanced training pipeline
Prepares data, models, and environment for production-quality training
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'tqdm', 'requests'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("   Please install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All requirements satisfied")
    return True

def check_gpu():
    """Check GPU availability"""
    print("🖥️  Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("⚠️  No GPU available, will use CPU (training will be slower)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_data():
    """Check if training data is available"""
    print("📊 Checking training data...")
    
    data_files = [
        'data/all_code_comments.json',
        'data/enhanced_all_pairs.json',
        'data/enhanced_python_pairs.json',
        'data/enhanced_javascript_pairs.json'
    ]
    
    existing_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
                print(f"✅ {file_path}: {len(data)} samples")
                existing_files.append(file_path)
        else:
            print(f"❌ {file_path}: Not found")
    
    if not existing_files:
        print("⚠️  No training data found!")
        print("   Please run data collection first:")
        print("   python tools/advanced_data_collector.py")
        return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'models/enhanced_comment',
        'models/enhanced_commit', 
        'logs',
        'data',
        'plots'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created: {directory}")

def process_data():
    """Process and prepare training data"""
    print("🔄 Processing training data...")
    
    try:
        from data.data_processor import EnhancedDataProcessor
        
        processor = EnhancedDataProcessor()
        
        # Find available data files
        input_files = []
        for file_path in ['data/enhanced_all_pairs.json', 'data/all_code_comments.json']:
            if os.path.exists(file_path):
                input_files.append(file_path)
        
        if not input_files:
            print("❌ No data files found for processing")
            return False
        
        # Process data
        processor.process_and_augment(
            input_files=input_files,
            output_file='data/production_training_data.json',
            target_size=5000  # Start with 5K samples
        )
        
        print("✅ Data processing completed")
        return True
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return False

def estimate_training_time():
    """Estimate training time based on hardware"""
    print("⏱️  Estimating training time...")
    
    try:
        import torch
        
        # Check hardware
        device = "GPU" if torch.cuda.is_available() else "CPU"
        
        if device == "GPU":
            gpu_name = torch.cuda.get_device_name(0)
            if "M1" in gpu_name or "M2" in gpu_name or "M3" in gpu_name or "M4" in gpu_name:
                print("🚀 Apple Silicon GPU detected - optimized for M-series chips")
                comment_time = "45-60 minutes"
                commit_time = "60-90 minutes"
            elif "RTX" in gpu_name or "GTX" in gpu_name:
                print("🚀 NVIDIA GPU detected")
                comment_time = "30-45 minutes"
                commit_time = "45-75 minutes"
            else:
                comment_time = "60-90 minutes"
                commit_time = "90-120 minutes"
        else:
            print("💻 CPU training (slower)")
            comment_time = "3-4 hours"
            commit_time = "4-6 hours"
        
        print(f"📊 Estimated training times:")
        print(f"   Comment Model (50M params): {comment_time}")
        print(f"   Commit Model (80M params): {commit_time}")
        print(f"   Total pipeline: ~2-3 hours on {device}")
        
    except ImportError:
        print("❌ Cannot estimate - PyTorch not available")

def create_training_script():
    """Create a simple training launcher script"""
    script_content = '''#!/usr/bin/env python3
"""
Enhanced Training Launcher
Run this script to start the enhanced training pipeline
"""

import sys
import os

def main():
    print("🚀 Starting Enhanced ML Training Pipeline")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists('data/production_training_data.json'):
        print("📊 Processing training data first...")
        os.system('python data/data_processor.py')
    
    # Start training
    print("\\n🧠 Starting enhanced model training...")
    os.system('python train_enhanced.py')

if __name__ == "__main__":
    main()
'''
    
    with open('backend/start_enhanced_training.py', 'w') as f:
        f.write(script_content)
    
    # Make executable
    os.chmod('backend/start_enhanced_training.py', 0o755)
    print("✅ Created training launcher: start_enhanced_training.py")

def main():
    """Main setup function"""
    print("🚀 Enhanced ML Training Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check GPU
    has_gpu = check_gpu()
    
    # Check data
    if not check_data():
        print("\n💡 To collect training data, run:")
        print("   python tools/advanced_data_collector.py")
        print("\n   Or use existing data with:")
        print("   python data/data_processor.py")
    
    # Setup directories
    setup_directories()
    
    # Process data if available
    if check_data():
        process_data()
    
    # Create training script
    create_training_script()
    
    # Estimate training time
    estimate_training_time()
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Ensure you have training data:")
    print("   python tools/advanced_data_collector.py  # Collect new data")
    print("   OR")
    print("   python data/data_processor.py            # Process existing data")
    print("\n2. Start enhanced training:")
    print("   python train_enhanced.py                 # Full training")
    print("   OR")
    print("   python start_enhanced_training.py        # Automated pipeline")
    print("\n3. Monitor training:")
    print("   - Check logs/ directory for training logs")
    print("   - Check models/ directory for saved models")
    print("   - Training plots will be saved automatically")
    
    if has_gpu:
        print("\n🚀 Your system is ready for fast GPU training!")
    else:
        print("\n💻 CPU training will be slower but still functional")

if __name__ == "__main__":
    main()