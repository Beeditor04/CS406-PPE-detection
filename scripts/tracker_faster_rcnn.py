import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
import cv2
import numpy as np
import time
from PIL import Image
from torchvision import models, transforms

from models.FASTER_RCNN import FASTER_RCNN
from utils.yaml_helper import read_yaml
from utils.function import non_max_suppression
from utils.function import draw_bbox, draw_fps
from utils.detect_css_violations import detect_css_violations
from trackers.byte_tracker import BYTETracker

# Argument parser
parser = argparse.ArgumentParser(description="Parser for Faster-RCNN tracking")
parser.add_argument("--weights", type=str, default="weights/faster-rcnn.pt", help="Path to pretrained weights")
parser.add_argument("--vid_dir", type=str, default="sample/video.mp4", help="Path to source video")
parser.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to class yaml file")
args = parser.parse_args()

# Load class names
yaml_class = read_yaml(args.yaml_class)
CLASS_NAMES = yaml_class["names"]

# Prepare output directory
if not os.path.exists("output/videos/"):
    if not os.path.exists("output/"):
        os.makedirs("output/")
    os.makedirs("output/videos")

# Setup video
OUTPUT_PATH = "output/videos/"
cap = cv2.VideoCapture(args.vid_dir)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# check if any output videos exist
num = 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_track_{num}_frcnn.mp4")):
    num += 1
out_track = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_track_{num}_frcnn.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_detect_{num}_frcnn.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_violate = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_violate_{num}_frcnn.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# Setup parameters
ASPECT_RATIO_THRESH = 0.6  # More condition for vertical box if you like
MIN_BOX_AREA = 100  # Minimum area of the tracking box to be considered
TRACK_THRESH = 0.5  # Tracking threshold
TRACK_BUFFER = 30  # Frame to keep track of the object
MATCH_THRESH = 0.85  # Matching threshold - similarity algorithm
FUSE_SCORE = False
DETECT_THRESH = 0.4

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def main():
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FASTER_RCNN(7)
    model.model.load_state_dict(torch.load(args.weights))
    model.model.to(device)
    model.eval()

    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    tracker_args = TrackerArgs(
        track_thresh=TRACK_THRESH,
        track_buffer=TRACK_BUFFER,
        match_thresh=MATCH_THRESH,
        fuse_score=FUSE_SCORE   
    )
    tracker = BYTETracker(tracker_args)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()

        #! Detection
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.model(image_tensor) # return xyxy format
            preds = [{k: v.to(device) for k, v in t.items()} for t in preds] # to device

        #!------------ Post-processing: filter low score, nms
        boxes = preds[0]['boxes']
        labels = preds[0]['labels'] - 1
        scores = preds[0]['scores']
        # print("Before NMS boxes:", boxes.shape)
        # print("Before NMS labels:", labels.shape)
        # print("Before NMS scores:", scores.shape)
        unique_labels = torch.unique(labels)
        final_boxes = []
        final_scores = []
        final_labels = []
        for label in unique_labels:
            class_mask = labels == label
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]

            keep = non_max_suppression(class_boxes, class_scores, iou_threshold=0.2)

            final_boxes.append(class_boxes[keep])
            final_scores.append(class_scores[keep])
            final_labels.append(torch.full((len(keep),), label, dtype=torch.int16))

        # Concatenate results
        if final_boxes:
            boxes = torch.cat(final_boxes)
            scores = torch.cat(final_scores)
            labels = torch.cat(final_labels)
        else:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            scores = torch.empty((0,), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int16)  
        # keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        # keep = non_max_suppression(boxes, scores, labels, iou_threshold=0.8)

        # print("After NMS boxes:", boxes.shape)
        # print("After NMS labels:", labels.shape)
        # print("After NMS scores:", scores.shape)
        
        #---
        per_detections = []
        obj_detections = [] 
        frame_detected = frame.copy()
        for i in range(boxes.shape[0]):
            if scores[i] < DETECT_THRESH:
                continue
            box = boxes[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            score = scores[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            draw_bbox(frame_detected, class_id, x1, y1, x2, y2, score, type='detect', class_names=CLASS_NAMES)
            # Detections bbox format for tracker
            if CLASS_NAMES[class_id] == "Person": # only track person
                per_detections.append([x1, y1, x2, y2, score])
                
            else:
                obj_detections.append([x1, y1, x2, y2, score, class_id])

        # Convert detections to numpy array
        if per_detections:
            per_detections = np.array(per_detections)
        else:
            per_detections = np.empty((0, 5))
        
        ##!----- CSS violation
        #! Update tracker with detections format
        online_targets = tracker.update(per_detections, [height, width], [height, width])
        online_targets = detect_css_violations(online_targets, obj_detections) #! CSS violation

        # Draw tracked objects
        frame_tracked = frame.copy()
        frame_violated = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            missing = t.missing if hasattr(t, 'missing') else 0
            if tlwh[2] * tlwh[3] > MIN_BOX_AREA:
                # Draw the bounding box
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, missing=missing, type='track')
                draw_bbox(frame_violated, tid, x1, y1, x2, y2, t.score, missing=missing, type='violate', class_names=CLASS_NAMES)

        # Calculate FPS
        process_time = time.time() - start_time
        fps = 1 / process_time

        # Save and display the frame
        draw_fps(cap, frame_detected, fps)
        draw_fps(cap, frame_tracked, fps)
        draw_fps(cap, frame_violated, fps)
        out_detect.write(frame_detected)
        out_track.write(frame_tracked)
        out_violate.write(frame_violated)

        frame_id += 1

    cap.release()
    out_detect.release()
    out_track.release()
    out_violate.release()
    print(f"Tracking results are saved in {OUTPUT_PATH}out_track_{num}_frcnn.mp4")

if __name__ == "__main__":
    main()