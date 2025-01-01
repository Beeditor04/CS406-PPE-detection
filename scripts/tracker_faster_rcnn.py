import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import argparse
import torch
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image

from models.FASTER_RCNN import FASTER_RCNN
from utils.yaml_helper import read_yaml
from test_another_name.byte_tracker import BYTETracker

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

# Check if any output videos exist
num_track = 1
num_detect = 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_track_{num_track}.mp4")):
    num_track += 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_detect_{num_detect}.mp4")):
    num_detect += 1
out_track = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_track_{num_track}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_detect_{num_detect}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# Setup parameters
ASPECT_RATIO_THRESH = 0.6  # More condition for vertical box if you like
MIN_BOX_AREA = 100  # Minimum area of the tracking box to be considered
TRACK_THRESH = 0.5  # Tracking threshold
TRACK_BUFFER = 30  # Frame to keep track of the object
MATCH_THRESH = 0.85  # Matching threshold - similarity algorithm
FUSE_SCORE = False
DETECT_THRESH = 0.5

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def draw_fps(frame):
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = ' FPS: ' + str(fps) + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
    cv2.putText(frame, fps, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, type='detect'):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if type == "detect":
        cv2.putText(frame, f"{CLASS_NAMES[id - 1]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if type == "track":
        cv2.putText(frame, f'ID: {id}, Score: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    tracking_results = []  # Store tracking results for eval, debug,...

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detection
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model.model(image_tensor)
            preds = [{k: v.to(device) for k, v in t.items()} for t in preds]

        boxes = preds[0]['boxes']
        labels = preds[0]['labels']
        scores = preds[0]['scores']

        detections = []
        frame_detected = frame.copy()
        for i in range(boxes.shape[0]):
            box = boxes[i].cpu().numpy()
            label = labels[i].cpu().numpy()
            score = scores[i].cpu().numpy()
            x1, y1, x2, y2 = map(int, box)
            class_id = int(label)
            
            # Detections bbox format for tracker
            detections.append([x1, y1, x2, y2, score])
            draw_bbox(frame_detected, class_id, x1, y1, x2, y2, score, type='detect')

        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))

        # Update tracker with detections format
        online_targets = tracker.update(detections, [height, width], [height, width])

        # Draw tracked objects
        frame_tracked = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            if tlwh[2] * tlwh[3] > MIN_BOX_AREA:
                # Save results
                tracking_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
                # Draw the bounding box
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, type='track')

        # Save and display the frame
        draw_fps(frame_detected)
        draw_fps(frame_tracked)
        out_detect.write(frame_detected)
        out_track.write(frame_tracked)

        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out_detect.release()
    out_track.release()
    cv2.destroyAllWindows()
    print(f"Tracking results are saved in {OUTPUT_PATH}out_track_{num_track}.mp4")

if __name__ == "__main__":
    main()