#!/usr/bin/env python3
"""
Resume Helmet Detection Model Training
Quick script to resume interrupted training
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
os.environ['OMP_NUM_THREADS'] = '2'

from ultralytics import YOLO

def resume_training():
    """Resume training from the last checkpoint"""
    
    # Check for the most recent checkpoint
    checkpoint_paths = [
        "runs/detect/helmet_robust/weights/last.pt",
        "runs/detect/helmet_detection/weights/last.pt",
        "runs/detect/helmet_detection4/weights/last.pt",
        "runs/detect/helmet_detection3/weights/last.pt",
        "runs/detect/helmet_detection2/weights/last.pt",
    ]
    
    checkpoint_path = None
    for path in checkpoint_paths:
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if not checkpoint_path:
        print("❌ No checkpoint found. Please start fresh training first.")
        print("Run: python train_robust.py")
        return
    
    print(f"🔄 Resuming training from: {checkpoint_path}")
    print("⏱️  This will complete the remaining epochs...")
    
    # Load model from checkpoint
    model = YOLO(checkpoint_path)
    
    # Resume training - YOLO will automatically continue from the last epoch
    results = model.train(resume=True)
    
    print("✅ Training resumed and completed!")
    
    # Save the final model
    best_model_path = results.save_dir / "weights" / "best.pt"
    final_model_path = "models/helmet_yolov11.pt"
    
    if best_model_path.exists():
        import shutil
        shutil.copy2(best_model_path, final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")

if __name__ == "__main__":
    resume_training()