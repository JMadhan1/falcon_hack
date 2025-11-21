"""
YOLO Training Script for Hackathon Dataset
Trains YOLOv8 model on educational object detection dataset
"""

from ultralytics import YOLO
import torch
import os
from pathlib import Path

def main():
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Using device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load a pretrained YOLOv8 model (nano version for faster training)
    # Options: yolov8n.pt (nano), yolov8s.pt (small), yolov8m.pt (medium), yolov8l.pt (large)
    model = YOLO('yolov8n.pt')  # Start with nano for speed
    
    print("\n📊 Dataset Configuration:")
    print("   Train: 1,767 images")
    print("   Val:   336 images")
    print("   Test:  1,408 images")
    print("   Classes: 7\n")
    
    # Training parameters
    results = model.train(
        data='data.yaml',           # path to dataset config
        epochs=100,                  # number of epochs
        imgsz=640,                   # image size
        batch=16,                    # batch size (adjust based on GPU memory)
        device=device,               # device to use
        workers=4,                   # number of worker threads
        patience=20,                 # early stopping patience
        save=True,                   # save checkpoints
        save_period=10,              # save checkpoint every N epochs
        project='runs/detect',       # project directory
        name='train',                # experiment name
        exist_ok=True,               # overwrite existing experiment
        pretrained=True,             # use pretrained weights
        optimizer='AdamW',           # optimizer (SGD, Adam, AdamW)
        verbose=True,                # verbose output
        seed=42,                     # random seed for reproducibility
        deterministic=True,          # deterministic mode
        single_cls=False,            # treat as single-class dataset
        rect=False,                  # rectangular training
        cos_lr=True,                 # cosine learning rate scheduler
        close_mosaic=10,             # disable mosaic augmentation for final epochs
        resume=False,                # resume from last checkpoint
        amp=True,                    # automatic mixed precision
        fraction=1.0,                # dataset fraction to train on
        profile=False,               # profile ONNX and TensorRT speeds
        freeze=None,                 # freeze layers (list or int)
        # Data augmentation
        hsv_h=0.015,                 # HSV-Hue augmentation
        hsv_s=0.7,                   # HSV-Saturation augmentation
        hsv_v=0.4,                   # HSV-Value augmentation
        degrees=0.0,                 # rotation (+/- deg)
        translate=0.1,               # translation (+/- fraction)
        scale=0.5,                   # scaling (+/- gain)
        shear=0.0,                   # shear (+/- deg)
        perspective=0.0,             # perspective (+/- fraction)
        flipud=0.0,                  # vertical flip probability
        fliplr=0.5,                  # horizontal flip probability
        mosaic=1.0,                  # mosaic augmentation probability
        mixup=0.0,                   # mixup augmentation probability
        copy_paste=0.0,              # copy-paste augmentation probability
    )
    
    print("\n✅ Training completed!")
    print(f"   Results saved to: {results.save_dir}")
    print(f"   Best weights: {results.save_dir}/weights/best.pt")
    print(f"   Last weights: {results.save_dir}/weights/last.pt")
    
    # Validate the model
    print("\n🔍 Validating model on validation set...")
    metrics = model.val()
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")

if __name__ == '__main__':
    main()
