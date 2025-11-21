import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io

app = Flask(__name__)

# Configuration
MODEL_PATH = 'runs/detect/train/weights/best.pt'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model Globaly
print("📦 Loading YOLOv8 Model...")
try:
    if os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ Model not found at {MODEL_PATH}")
        model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Read image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        img_np = np.array(img)

        # Run inference
        results = model.predict(img_np, conf=0.25)
        
        # Process results
        result = results[0]
        
        # Get detections
        detections = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = result.names[cls_id]
            
            detections.append({
                'class': name,
                'confidence': f"{conf:.1%}",
                'bbox': box.xyxy[0].tolist()
            })

        # Plot results on image
        res_plotted = result.plot()
        res_img = Image.fromarray(res_plotted[..., ::-1]) # RGB to BGR fix if needed, but plot() usually returns BGR for cv2 or RGB? Ultralytics plot() returns BGR numpy array.
        # Actually ultralytics plot() returns a numpy array in BGR (OpenCV format). 
        # We need to convert it to RGB for PIL or just encode it as JPEG.
        
        # Convert BGR to RGB for PIL
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_pil = Image.fromarray(res_plotted_rgb)
        
        # Save to buffer
        buffered = io.BytesIO()
        res_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': img_str,
            'detections': detections
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
