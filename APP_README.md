# Bonus Application: Space Station Safety Detector Web App

## Overview
This is a **Streamlit-based web application** that demonstrates the trained YOLOv8 object detection model for identifying safety equipment in space station environments.

## Features

### ✨ Core Functionality
- **Image Upload Detection:** Upload space station images and get real-time object detection
- **Interactive Controls:** Adjust confidence threshold and IoU settings
- **Visual Results:** Side-by-side comparison of original and annotated images
- **Detection Statistics:** Detailed metrics including confidence scores and bounding boxes
- **Class Information:** Display of all 7 detectable safety objects

### 📊 Detection Capabilities
The app detects the following safety equipment:
1. **OxygenTank** - Life support systems
2. **NitrogenTank** - Atmospheric regulation
3. **FirstAidBox** - Emergency medical supplies
4. **FireAlarm** - Fire detection systems
5. **SafetySwitchPanel** - Emergency control panels
6. **EmergencyPhone** - Communication devices
7. **FireExtinguisher** - Fire suppression equipment

## Installation & Setup

### Prerequisites
- Python 3.10+
- Trained model weights at `runs/detect/train/weights/best.pt`

### Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `streamlit` - Web app framework
- `ultralytics` - YOLOv8 inference
- `opencv-python` - Image processing
- `pillow` - Image handling

## Running the Application

### Start the Web App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### Using the App

1. **Upload an Image**
   - Click "Browse files" in the Upload Image tab
   - Select a space station image (JPG, JPEG, or PNG)
   - Wait for detection results

2. **Adjust Settings** (Optional)
   - Use the sidebar sliders to adjust:
     - **Confidence Threshold:** Minimum confidence for detections (default: 0.45)
     - **IoU Threshold:** NMS threshold for overlapping boxes (default: 0.45)

3. **View Results**
   - Original image on the left
   - Annotated image with bounding boxes on the right
   - Detection statistics below (total detections, confidence scores)
   - Detailed table of all detected objects

## Application Architecture

```
app.py
├── Page Configuration
├── Custom CSS Styling
├── Model Loading (@st.cache_resource)
├── Sidebar
│   ├── Detection Settings (sliders)
│   ├── Model Information
│   └── Class List
└── Main Tabs
    ├── Upload Image (detection interface)
    ├── Webcam (planned feature)
    └── About (project information)
```

## Falcon Integration for Continuous Improvement

See [`FALCON_INTEGRATION_PLAN.md`](FALCON_INTEGRATION_PLAN.md) for a comprehensive guide on how to use Duality AI's Falcon platform to continuously improve this model.

### Key Benefits:
- **Automated Data Generation:** Create synthetic training data for failure cases
- **No Manual Labeling:** Falcon auto-generates ground truth labels
- **Safe Testing:** Simulate dangerous scenarios without risk
- **Continuous Learning:** Feedback loop for perpetual model improvement

### Quick Summary:
1. Deploy model → Collect failure cases
2. Recreate failures in Falcon → Generate synthetic data
3. Retrain model → Validate improvements
4. Deploy updated model → Repeat

## Future Enhancements

### Planned Features:
- [ ] **Real-time Webcam Detection** - Live video stream processing
- [ ] **Batch Processing** - Upload multiple images at once
- [ ] **Detection History** - Track and compare results over time
- [ ] **Alert System** - Notifications for critical equipment detection
- [ ] **Export Results** - Download annotated images and CSV reports
- [ ] **Mobile App** - iOS/Android version using TensorFlow Lite

### Model Improvements:
- [ ] Upgrade to YOLOv8s/m for higher accuracy
- [ ] Add instance segmentation for precise boundaries
- [ ] Implement tracking for video sequences
- [ ] Multi-language support for international crews

## Technical Specifications

| Metric | Value |
|--------|-------|
| Framework | Streamlit 1.28+ |
| Model | YOLOv8n |
| Inference Speed | ~45ms/image (GPU) |
| Model Size | 6.2 MB |
| Supported Formats | JPG, JPEG, PNG |
| Browser Support | Chrome, Firefox, Safari, Edge |

## Troubleshooting

### Model Not Found Error
```
❌ Model not found! Please train the model first using `python train.py`
```
**Solution:** Ensure you have trained the model and `runs/detect/train/weights/best.pt` exists.

### Slow Inference
**Solution:** 
- Use a GPU-enabled machine
- Reduce image resolution before upload
- Lower confidence threshold to reduce processing

### Port Already in Use
```
OSError: [Errno 98] Address already in use
```
**Solution:** 
```bash
streamlit run app.py --server.port 8502
```

## Demo Video

A video demonstration of the application is included in the submission package showing:
- Image upload workflow
- Real-time detection
- Confidence threshold adjustment
- Detection statistics

## Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Cloud Deployment (Streamlit Cloud)
1. Push code to GitHub
2. Connect repository to [Streamlit Cloud](https://streamlit.io/cloud)
3. Deploy with one click

### Docker Deployment
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## License
This application is part of the Duality AI Space Station Challenge #2 submission.

## Team
**AI LONE STARS**

## Acknowledgments
- **Duality AI** for the Falcon platform and synthetic dataset
- **Ultralytics** for the YOLOv8 framework
- **Streamlit** for the web app framework

---

**For questions or issues, please refer to the main [README.md](README.md) or [HACKATHON_REPORT.md](HACKATHON_REPORT.md)**
