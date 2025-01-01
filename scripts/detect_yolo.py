import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse

import cv2
from ultralytics import YOLO

from utils.yaml_helper import read_yaml
from utils.function import non_max_suppression

# Argument parser
parser = argparse.ArgumentParser(description="Parser for YOLO inference")
parser.add_argument("--weights", type=str, default="weights/yolo.pt", help="Path to pretrained weights")
parser.add_argument("--img_path", type=str, default="sample/1.jpg", help="Path to source image")
parser.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to class yaml file")

args = parser.parse_args()
yaml_class = read_yaml(args.yaml_class)
CLASS_NAMES = yaml_class["names"]
DETECT_THRESH = 0.5

def draw_bboxes(image, detections, class_names):
    for detection in detections:
        for box in detection.boxes:
            label, conf, bbox = int(box.cls[0]), float(box.conf[0]), box.xyxy.tolist()[0]
            x1, y1, x2, y2 = map(int, bbox)
            class_id = int(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(image, f"{class_names[class_id]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def inference(weights, img_path):
    model = YOLO(weights, DETECT_THRESH)

    # load img
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Image file '{img_path}' not found.")
    
    # Perform inference
    results = model(image)
    image_with_bboxes = draw_bboxes(image, results, CLASS_NAMES)

    # Save the image
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = 1
    while os.path.exists(os.path.join(output_dir, f"inference-{num}-yolo.jpg")):
        num += 1
    output_path = os.path.join(output_dir, f"inference-{num}-yolo.jpg")
    cv2.imwrite(output_path, image_with_bboxes)
    print(f"Saved inference result to {output_path}")

if __name__ == "__main__":
    inference(args.weights, args.img_path)