#!/usr/bin/env python3
"""
Helmet Detection Model Training Script
Trains a YOLOv11 model on the helmet detection dataset
"""

import os
from ultralytics import YOLO
import yaml
from pathlib import Path

def train_helmet_model(resume=False):
    """Train YOLOv11 model on helmet detection dataset"""
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Check for existing checkpoint
    checkpoint_path = "runs/detect/helmet_detection/weights/last.pt"
    
    if resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming training from checkpoint: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        print("🆕 Starting fresh training...")
        # Load YOLOv11 model (you can use different sizes: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)
        model = YOLO('yolo11n.pt')  # Using nano for faster training, change to 'yolo11s.pt' for better accuracy
    
    # Update data.yaml paths to be absolute
    data_yaml_path = "helmet_dataset/data.yaml"
    
    # Read and update the data.yaml file
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths to be relative to the data.yaml file location
    base_path = Path("helmet_dataset")
    data_config['train'] = str(base_path / "train" / "images")
    data_config['val'] = str(base_path / "valid" / "images") 
    data_config['test'] = str(base_path / "test" / "images")
    
    # Write updated config
    updated_yaml_path = "helmet_dataset/data_updated.yaml"
    with open(updated_yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    print("🚀 Starting helmet detection model training...")
    print(f"📊 Dataset classes: {data_config['names']}")
    print(f"📁 Training images: {data_config['train']}")
    print(f"📁 Validation images: {data_config['val']}")
    
    # Train the model
    results = model.train(
        data=updated_yaml_path,
        epochs=20,   # Reduced for faster training
        imgsz=640,   # Image size
        batch=16,    # Batch size (adjust based on your GPU memory)
        name='helmet_detection',
        patience=5,  # Early stopping patience (reduced for 20 epochs)
        save=True,
        plots=True,
        device='cpu',  # Change to 'cuda' if you have GPU
        workers=4,
        cache=False,  # Set to True if you have enough RAM
        resume=resume,  # Resume from checkpoint if available
    )
    
    # Save the best model to our models directory
    best_model_path = results.save_dir / "weights" / "best.pt"
    final_model_path = "models/helmet_yolov11.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"✅ Model saved to: {final_model_path}")
    
    print("🎉 Training completed!")
    print(f"📈 Results saved in: {results.save_dir}")
    
    return results

def validate_dataset():
    """Validate the dataset structure and count samples"""
    dataset_path = Path("helmet_dataset")
    
    for split in ['train', 'valid', 'test']:
        images_path = dataset_path / split / "images"
        labels_path = dataset_path / split / "labels"
        
        if images_path.exists() and labels_path.exists():
            image_count = len(list(images_path.glob("*.jpg"))) + len(list(images_path.glob("*.png")))
            label_count = len(list(labels_path.glob("*.txt")))
            print(f"📊 {split.upper()} set: {image_count} images, {label_count} labels")
        else:
            print(f"❌ {split.upper()} set: Missing images or labels directory")

if __name__ == "__main__":
    print("🔍 Validating dataset...")
    validate_dataset()
    
    # Check for existing checkpoint
    checkpoint_path = "runs/detect/helmet_detection/weights/last.pt"
    has_checkpoint = os.path.exists(checkpoint_path)
    
    if has_checkpoint:
        print(f"\n🔄 Found existing checkpoint: {checkpoint_path}")
        print("Options:")
        print("1. Resume training from checkpoint (r)")
        print("2. Start fresh training (f)")
        print("3. Cancel (n)")
        response = input("Choose option (r/f/n): ")
        
        if response.lower() in ['r', 'resume']:
            train_helmet_model(resume=True)
        elif response.lower() in ['f', 'fresh']:
            train_helmet_model(resume=False)
        else:
            print("Training cancelled.")
    else:
        print("\n" + "="*50)
        response = input("Start training? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            train_helmet_model(resume=False)
        else:
            print("Training cancelled.")