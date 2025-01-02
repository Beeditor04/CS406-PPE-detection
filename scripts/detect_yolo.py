import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image

from utils.yaml_helper import read_yaml
from utils.function import non_max_suppression
from utils.function import draw_bbox
from utils.detect_css_violations import detect_css_violations
from utils.detect_css_violations import STrack
# Argument parser
parser = argparse.ArgumentParser(description="Parser for YOLO inference")
parser.add_argument("--weights", type=str, default="weights/yolo.pt", help="Path to pretrained weights")
parser.add_argument("--img_path", type=str, default="sample/1.jpg", help="Path to source image")
parser.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to class yaml file")

args = parser.parse_args()
yaml_class = read_yaml(args.yaml_class)
CLASS_NAMES = yaml_class["names"]
DETECT_THRESH = 0.4

def inference(device, weights, img_path):
    model = YOLO(weights, DETECT_THRESH)
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    # load img
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    # Perform inference
    with torch.no_grad():
        image = Image.open(img_path).convert('RGB') # for faster_rcnn
        image_tensor = transform(image).unsqueeze(0).to(device)
        detect_results = model(image_tensor)
        image = cv2.imread(img_path) # back to cv2
        image = cv2.resize(image, (640, 640))
    
    #!------------ Post-processing: filter low score, nms
    boxes = []
    scores = []
    labels = []
    for result in detect_results:
        for box in result.boxes:
            label, conf, bbox = int(box.cls[0]), float(box.conf[0]), box.xyxy.tolist()[0]
            if conf >= DETECT_THRESH:
                boxes.append(bbox)
                scores.append(conf)
                labels.append(label)

    keep = non_max_suppression(boxes, scores, iou_threshold=0.7)
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    labels = [labels[i] for i in keep]

    #----
    per_detections = []
    obj_detections = []
    image_detection = image.copy()
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        class_id = int(labels[i])
        score = float(scores[i])
        # Detections bbox format for tracker
        if CLASS_NAMES[class_id] == "Person": # only track person
            per_detections.append([x1, y1, x2, y2, score])
            
        else:
            obj_detections.append([x1, y1, x2, y2, score, class_id])
        draw_bbox(image_detection, CLASS_NAMES[class_id], x1, y1, x2, y2, score, type='detect')
    
    ## construct to detect_css_violation format
    online_targets = []
    for i, t in enumerate(per_detections):
        tlwh = [t[0], t[1], t[2]-t[0], t[3]-t[1]] # xyxy to tlwh
        online_targets.append(STrack(tlwh, t[4], i))
    
    # Convert per_detections to numpy array
    if per_detections:
        per_detections = np.array(per_detections)
    else:
        per_detections = np.empty((0, 5))

    ## CSS violation
    online_targets = detect_css_violations(online_targets, obj_detections) #! CSS violation

    image_violate = image.copy()
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        print("Missing:", t.missing)
        x1, y1, w, h = map(int, tlwh)
        x2, y2 = x1 + w, y1 + h
        draw_bbox(image_violate, tid, x1, y1, x2, y2, t.score, missing=t.missing, type='violate', class_names=CLASS_NAMES)

    # Save the image
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = 1

    while os.path.exists(os.path.join(output_dir, f"inference-{num}-yolo.jpg")):
        num += 1
    output_path = os.path.join(output_dir, f"inference-{num}-yolo.jpg")
    cv2.imwrite(output_path, image_detection)
    print(f"Saved inference result to {output_path}") 

    output_path = os.path.join(output_dir, f"inference-{num}-yolo-violate.jpg")
    cv2.imwrite(output_path, image_violate)
    print(f"Saved violate detection result to {output_path}") 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(device, args.weights, args.img_path)