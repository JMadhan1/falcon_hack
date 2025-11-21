"""
YOLO Inference Script for Test Dataset
Runs predictions on test images and saves results
"""

from ultralytics import YOLO
import os
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

def main():
    # Load the trained model
    model_path = 'runs/detect/train/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("   Please train the model first using train.py")
        return
    
    print(f"📦 Loading model from {model_path}")
    model = YOLO(model_path)
    
    # Test images directory
    test_dir = Path('dataset/test/images')
    output_dir = Path('runs/detect/test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔍 Running inference on test dataset...")
    print(f"   Test images: {test_dir}")
    print(f"   Output directory: {output_dir}")
    
    # Run inference on test dataset
    results = model.predict(
        source=str(test_dir),
        save=True,                    # save images with predictions
        save_txt=True,                # save results as .txt
        save_conf=True,               # save confidences in .txt
        project='runs/detect',
        name='test',
        exist_ok=True,
        conf=0.25,                    # confidence threshold
        iou=0.45,                     # NMS IoU threshold
        imgsz=640,                    # image size
        device='cuda' if model.device.type == 'cuda' else 'cpu',
        verbose=True,
        stream=True,                  # stream results
    )
    
    # Process and save results
    print("\n📊 Processing results...")
    all_predictions = []
    
    for result in tqdm(results, desc="Processing images"):
        img_name = Path(result.path).name
        
        # Extract predictions
        boxes = result.boxes
        for box in boxes:
            pred = {
                'image': img_name,
                'class_id': int(box.cls[0]),
                'class_name': result.names[int(box.cls[0])],
                'confidence': float(box.conf[0]),
                'x_center': float(box.xywhn[0][0]),
                'y_center': float(box.xywhn[0][1]),
                'width': float(box.xywhn[0][2]),
                'height': float(box.xywhn[0][3]),
            }
            all_predictions.append(pred)
    
    # Save predictions to CSV
    if all_predictions:
        df = pd.DataFrame(all_predictions)
        csv_path = output_dir / 'predictions.csv'
        df.to_csv(csv_path, index=False)
        print(f"\n✅ Predictions saved to {csv_path}")
        
        # Print summary statistics
        print("\n📈 Prediction Summary:")
        print(f"   Total images: {len(df['image'].unique())}")
        print(f"   Total detections: {len(df)}")
        print(f"\n   Detections per class:")
        for class_name, count in df['class_name'].value_counts().items():
            print(f"      {class_name}: {count}")
        print(f"\n   Average confidence: {df['confidence'].mean():.4f}")
        print(f"   Min confidence: {df['confidence'].min():.4f}")
        print(f"   Max confidence: {df['confidence'].max():.4f}")
    else:
        print("\n⚠️  No predictions found!")
    
    print(f"\n✅ Inference completed!")
    print(f"   Annotated images saved to: {output_dir}")

if __name__ == '__main__':
    main()
