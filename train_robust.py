#!/usr/bin/env python3
"""
Robust Helmet Detection Training with Error Handling
"""

import os
import sys
from ultralytics import YOLO
import yaml
from pathlib import Path
import torch

def setup_environment():
    """Setup environment variables for stable training"""
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    os.environ['OMP_NUM_THREADS'] = '2'  # Limit threads to prevent overload
    
def train_helmet_model():
    """Train with robust settings"""
    
    try:
        setup_environment()
        
        # Create models directory
        os.makedirs("models", exist_ok=True)
        
        print("🚀 Starting robust helmet detection training...")
        print(f"💻 Device: CPU (Intel Core i3-6100U)")
        print(f"🧠 Available RAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3 if torch.cuda.is_available() else 'CPU only'}")
        
        # Load YOLOv11 model
        model = YOLO('yolo11n.pt')
        
        # Update data.yaml paths
        data_yaml_path = "helmet_dataset/data.yaml"
        
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Use absolute paths
        current_dir = Path.cwd()
        data_config['train'] = str(current_dir / "helmet_dataset" / "train" / "images")
        data_config['val'] = str(current_dir / "helmet_dataset" / "valid" / "images") 
        data_config['test'] = str(current_dir / "helmet_dataset" / "test" / "images")
        
        updated_yaml_path = "helmet_dataset/data_robust.yaml"
        with open(updated_yaml_path, 'w') as f:
            yaml.dump(data_config, f)
        
        print(f"📊 Classes: {data_config['names']}")
        print(f"📁 Training images: 3,299")
        print(f"📁 Validation images: 441")
        print("⚙️  Using conservative settings for stability...")
        
        # Conservative training settings for CPU
        results = model.train(
            data=updated_yaml_path,
            epochs=10,        # Reduced epochs for testing
            imgsz=416,        # Smaller image size for CPU
            batch=8,          # Smaller batch size
            name='helmet_robust',
            patience=3,       # Early stopping
            save=True,
            plots=True,
            device='cpu',
            workers=2,        # Reduced workers
            cache=False,
            verbose=True,
            amp=False,        # Disable mixed precision for CPU
            optimizer='SGD',  # More stable optimizer
            lr0=0.001,        # Lower learning rate
            momentum=0.9,
            weight_decay=0.0005,
            warmup_epochs=1,  # Reduced warmup
        )
        
        print("✅ Training completed!")
        
        # Save the final model
        best_model_path = results.save_dir / "weights" / "best.pt"
        final_model_path = "models/helmet_yolov11.pt"
        
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"✅ Model saved to: {final_model_path}")
            return True
        else:
            print("⚠️  No best model found, checking for last model...")
            last_model_path = results.save_dir / "weights" / "last.pt"
            if last_model_path.exists():
                import shutil
                shutil.copy2(last_model_path, final_model_path)
                print(f"✅ Last model saved to: {final_model_path}")
                return True
        
        return False
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        print("💡 Suggestions:")
        print("   - Close other applications to free memory")
        print("   - Try even smaller batch size (batch=4)")
        print("   - Use smaller image size (imgsz=320)")
        return False

if __name__ == "__main__":
    success = train_helmet_model()
    if success:
        print("\n🎉 Training successful! You can now use the model.")
    else:
        print("\n❌ Training failed. Check the suggestions above.")