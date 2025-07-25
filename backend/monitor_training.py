#!/usr/bin/env python3
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
        
        print("\nPress Ctrl+C to exit")
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_training()
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
