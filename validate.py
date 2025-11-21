"""
Validate trained YOLO model on validation dataset
"""

from ultralytics import YOLO
import os

def main():
    # Load the trained model
    model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first using train.py")
        return
    
    print(f"📦 Loading model from {model_path}")
    model = YOLO(model_path)
    
    print("\n🔍 Validating model on validation dataset...")
    
    # Validate the model
    metrics = model.val(
        data='data.yaml',
        split='val',
        imgsz=640,
        batch=16,
        conf=0.25,
        iou=0.45,
        device='cuda' if model.device.type == 'cuda' else 'cpu',
        save_json=True,
        save_hybrid=False,
        plots=True,
        verbose=True,
    )
    
    print(f"\n📈 Validation Metrics:")
    print(f"   mAP50: {metrics.box.map50:.4f}")
    print(f"   mAP50-95: {metrics.box.map:.4f}")
    print(f"   Precision: {metrics.box.mp:.4f}")
    print(f"   Recall: {metrics.box.mr:.4f}")
    
    # Per-class metrics
    print(f"\n📊 Per-Class Metrics:")
    for i, class_name in enumerate(model.names.values()):
        if i < len(metrics.box.ap_class_index):
            print(f"   {class_name}:")
            print(f"      Precision: {metrics.box.p[i]:.4f}")
            print(f"      Recall: {metrics.box.r[i]:.4f}")
            print(f"      mAP50: {metrics.box.ap50[i]:.4f}")
    
    print("\n✅ Validation completed!")

if __name__ == '__main__':
    main()
