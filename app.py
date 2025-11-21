import os
import cv2
import numpy as np
import base64
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import json
from datetime import datetime

app = Flask(__name__)

# Configuration
MODEL_PATH = 'runs/detect/train/weights/best.pt'
UPLOAD_FOLDER = 'static/uploads'
FEEDBACK_FOLDER = 'feedback_data'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

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
        result = results[0]
        
        # Process detections
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

        # Convert plot to base64
        res_plotted = result.plot()
        res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
        res_pil = Image.fromarray(res_plotted_rgb)
        
        buffered = io.BytesIO()
        res_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Save original image temporarily for feedback reference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_query.jpg"
        img.save(os.path.join(UPLOAD_FOLDER, filename))

        return jsonify({
            'success': True,
            'image': img_str,
            'detections': detections,
            'image_id': filename
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """
    Endpoint to collect user feedback for Continuous Learning.
    This simulates sending data back to Falcon for retraining.
    """
    try:
        data = request.json
        image_id = data.get('image_id')
        correct_class = data.get('correct_class')
        notes = data.get('notes')
        
        # In a real scenario, this would upload to S3 or Duality's API
        # Here we save metadata locally
        feedback_entry = {
            'image_id': image_id,
            'timestamp': datetime.now().isoformat(),
            'user_correction': correct_class,
            'notes': notes,
            'status': 'queued_for_falcon_retraining'
        }
        
        with open(os.path.join(FEEDBACK_FOLDER, 'feedback_log.json'), 'a') as f:
            f.write(json.dumps(feedback_entry) + '\n')
            
        return jsonify({'success': True, 'message': 'Feedback received. Data queued for Falcon retraining loop.'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
