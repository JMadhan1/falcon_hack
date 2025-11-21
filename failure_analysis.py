import os
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from tqdm import tqdm
import shutil

def xywhn2xyxy(x, w=640, h=640):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2]
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2)  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2)  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2)  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2)  # bottom right y
    return y

def compute_iou(box1, box2):
    # box1: [x1, y1, x2, y2], box2: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def main():
    # Setup paths
    base_dir = Path('.')
    val_img_dir = base_dir / 'dataset/val/images'
    val_label_dir = base_dir / 'dataset/val/labels'
    output_dir = base_dir / 'runs/detect/failure_analysis'
    model_path = base_dir / 'runs/detect/train/weights/best.pt'

    # Create output directories
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'false_positives').mkdir()
    (output_dir / 'false_negatives').mkdir()
    (output_dir / 'low_iou').mkdir()

    # Load model
    print(f"📦 Loading model from {model_path}")
    model = YOLO(model_path)
    names = model.names

    print(f"🔍 Analyzing validation set for failures...")
    
    # Get list of validation images
    img_files = list(val_img_dir.glob('*.jpg'))
    
    failures_count = 0
    
    for img_file in tqdm(img_files):
        label_file = val_label_dir / img_file.with_suffix('.txt').name
        
        # Read ground truth
        gt_boxes = []
        gt_classes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    # Convert normalized xywh to xyxy
                    bbox = [float(x) for x in parts[1:]]
                    # We'll convert to pixel coords later when we have image size
                    gt_boxes.append(bbox)
                    gt_classes.append(cls_id)

        # Run inference
        results = model(img_file, verbose=False)[0]
        img_h, img_w = results.orig_shape
        
        # Convert GT to pixels
        gt_boxes_px = []
        if gt_boxes:
            gt_boxes_norm = np.array(gt_boxes)
            gt_boxes_px = xywhn2xyxy(gt_boxes_norm, w=img_w, h=img_h)

        # Process predictions
        pred_boxes = results.boxes.xyxy.cpu().numpy()
        pred_classes = results.boxes.cls.cpu().numpy().astype(int)
        pred_confs = results.boxes.conf.cpu().numpy()

        # Match predictions to GT
        matched_gt = set()
        
        # Load image for annotation
        img = cv2.imread(str(img_file))
        has_failure = False
        
        # Check for False Positives and Low IoU
        for i, pred_box in enumerate(pred_boxes):
            best_iou = 0
            best_gt_idx = -1
            
            for j, gt_box in enumerate(gt_boxes_px):
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
            
            cls_name = names[pred_classes[i]]
            
            if best_iou < 0.5:
                # False Positive (or very poor localization)
                has_failure = True
                cv2.rectangle(img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0, 0, 255), 2)
                cv2.putText(img, f"FP {cls_name} {pred_confs[i]:.2f}", (int(pred_box[0]), int(pred_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imwrite(str(output_dir / 'false_positives' / f"fp_{img_file.name}"), img)
            elif pred_classes[i] != gt_classes[best_gt_idx]:
                # Misclassification (High IoU but wrong class)
                has_failure = True
                cv2.rectangle(img, (int(pred_box[0]), int(pred_box[1])), (int(pred_box[2]), int(pred_box[3])), (0, 165, 255), 2)
                cv2.putText(img, f"WrongCls {cls_name} vs {names[gt_classes[best_gt_idx]]}", (int(pred_box[0]), int(pred_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                cv2.imwrite(str(output_dir / 'false_positives' / f"misclass_{img_file.name}"), img)
            else:
                matched_gt.add(best_gt_idx)

        # Check for False Negatives (Missed GT)
        for j, gt_box in enumerate(gt_boxes_px):
            if j not in matched_gt:
                has_failure = True
                cls_name = names[gt_classes[j]]
                cv2.rectangle(img, (int(gt_box[0]), int(gt_box[1])), (int(gt_box[2]), int(gt_box[3])), (0, 255, 0), 2)
                cv2.putText(img, f"FN {cls_name}", (int(gt_box[0]), int(gt_box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imwrite(str(output_dir / 'false_negatives' / f"fn_{img_file.name}"), img)

        if has_failure:
            failures_count += 1
            if failures_count > 50: # Limit to 50 failure examples to save time/space
                break

    print(f"\n✅ Analysis complete!")
    print(f"   Failure cases saved to: {output_dir}")

if __name__ == '__main__':
    main()
