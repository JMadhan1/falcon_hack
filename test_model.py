from ultralytics import YOLO
from pathlib import Path

# Load model
model_path = Path('runs/detect/train/weights/best.pt')
print(f"Loading model from: {model_path}")
print(f"Model exists: {model_path.exists()}")

model = YOLO(model_path)
print(f"Model loaded successfully!")
print(f"Model names: {model.names}")

# Test on a sample image
test_img = Path('dataset/test/images/000000000_vdark_clutter.png')
if test_img.exists():
    print(f"\nTesting on: {test_img}")
    results = model.predict(source=str(test_img), conf=0.25, verbose=True)
    print(f"Number of detections: {len(results[0].boxes)}")
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - {model.names[cls_id]}: {conf:.2f}")
else:
    print(f"Test image not found: {test_img}")
