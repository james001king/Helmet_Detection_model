#!/usr/bin/env python3
"""
Auto-start helmet detection training
"""

import os
from ultralytics import YOLO
import yaml
from pathlib import Path

def auto_train():
    """Automatically start training without user input"""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    print("🚀 Auto-starting helmet detection model training...")
    
    # Load YOLOv11 model
    model = YOLO('yolo11n.pt')
    
    # Update data.yaml paths
    data_yaml_path = "helmet_dataset/data.yaml"
    
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Use absolute paths to avoid confusion
    current_dir = Path.cwd()
    data_config['train'] = str(current_dir / "helmet_dataset" / "train" / "images")
    data_config['val'] = str(current_dir / "helmet_dataset" / "valid" / "images") 
    data_config['test'] = str(current_dir / "helmet_dataset" / "test" / "images")
    
    updated_yaml_path = "helmet_dataset/data_updated.yaml"
    with open(updated_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"📊 Dataset classes: {data_config['names']}")
    print(f"📁 Training on {len(list((current_dir / 'helmet_dataset' / 'train' / 'images').glob('*')))} images")
    print("⏱️  Estimated time: 25-40 minutes for 20 epochs")
    print("🔄 Training will auto-save checkpoints every epoch")
    
    # Start training
    results = model.train(
        data=updated_yaml_path,
        epochs=20,
        imgsz=640,
        batch=16,
        name='helmet_detection',
        patience=5,
        save=True,
        plots=True,
        device='cpu',
        workers=4,
        cache=False,
    )
    
    # Save final model
    best_model_path = results.save_dir / "weights" / "best.pt"
    final_model_path = "models/helmet_yolov11.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")
    
    print("🎉 Training completed successfully!")
    print(f"📈 Results saved in: {results.save_dir}")
    
    return results

if __name__ == "__main__":
    auto_train()