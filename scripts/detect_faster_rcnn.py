import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
import argparse
import torch
from torchvision import models, transforms
import cv2
import os
import numpy as np
from PIL import Image

from models.FASTER_RCNN import FASTER_RCNN
from utils.function import non_max_suppression
from utils.yaml_helper import read_yaml

DETECT_THRESH = 0.3
parse  = argparse.ArgumentParser(description="Parser for Faster-RCNN inference")
parse.add_argument("--weights", type=str, default="weights/faster-rcnn.pt", help="Path to pretrain weights")
parse.add_argument("--img_path", type=str, default="sample/1.jpg", help="Path to source image")
parse.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to class yaml")
args = parse.parse_args()

yaml_class = read_yaml(args.yaml_class)
CLASS_NAMES = yaml_class["names"]
def show_preds(image, image_tensor, preds, output_dir="output", threshold=0.3, class_names=[]):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find the next available file number
    num = 1
    while os.path.exists(os.path.join(output_dir, f"inference-{num}-frcnn.jpg")):
        num += 1

    boxes = preds[0]['boxes']
    labels = preds[0]['labels']
    scores = preds[0]['scores']

    # post-processing
    keep = non_max_suppression(boxes, scores, iou_threshold=0.3)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for i in range(boxes.shape[0]):
        if scores[i] < threshold:
            continue
        box = boxes[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        score = scores[i].cpu().numpy()
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (255, 254, 0), 
            2
        )
        cv2.putText(
            image,
            f"{class_names[label - 1]} {score:.2f}", # increment label by 1 to ignore background, so -1 to get the correct class name
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        # Save the image
    output_path = os.path.join(output_dir, f"inference-{num}-frcnn.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved inference result to {output_path}")

def inference(device, pretrain="weights/faster-rcnn.pt", src_path="sample/1.jpg"):
    model = FASTER_RCNN(7)
    model.model.load_state_dict(torch.load(pretrain))
    model.model.to(device)
    model.eval()

    threshold = 0.5

    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        normalize
    ])
    
    with torch.no_grad():
        image = Image.open(src_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        image = image.resize((640, 640))
        preds = model.model(image_tensor)
        preds = [{k: v.to(device) for k, v in t.items()} for t in preds]

    show_preds(image, image_tensor, preds, threshold=threshold, class_names=CLASS_NAMES)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(device, pretrain=args.weights, src_path=args.img_path)