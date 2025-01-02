import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parsers.parser_tracker import parse_args
from utils.yaml_helper import read_yaml
from utils.function import non_max_suppression
from utils.function import draw_bbox, draw_fps
from utils.detect_css_violations import detect_css_violations
from trackers.byte_tracker import BYTETracker

import torch
import cv2
import numpy as np
from ultralytics import YOLO

# SETUP parse
args = parse_args()

# yaml class
yaml_class = read_yaml(args.yaml_class)
CLASS_NAMES = yaml_class["names"]

# prepare output dir
if not os.path.exists("output/videos/"):
    if not os.path.exists("output/"):
        os.makedirs("output/")
    os.makedirs("output/videos")

# SETUP video
OUTPUT_PATH = "output/videos/"
cap = cv2.VideoCapture(args.vid_dir)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# check if any output videos exist
num = 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_track_{num}_yolo.mp4")):
    num += 1
out_track = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_track_{num}_yolo.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_detect_{num}_yolo.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_violate = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_violate_{num}_yolo.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

# setup params
ASPECT_RATIO_THRESH = 0.6  # more condition for vertical box if you like
MIN_BOX_AREA = 100 # minimum area of the trcking box to be considered
TRACK_THRESH= 0.5 # tracking threshold
TRACK_BUFFER= 30 # frame to keep track of the object
MATCH_THRESH= 0.85 # matching threshold - similarity alogrithm
FUSE_SCORE= False 
DETECT_THRESH = 0.5

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False


# For future these above functions can be moved to a separate file
#================================================================================================
def main():
    weights = args.weights
    # model = YOLO_MODEL(weights)
    model = YOLO(weights, DETECT_THRESH)
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
        #!---------- Detect
        # detect_results = model.detect(frame) # return xyxy format
        detect_results = model(frame) # return xyxy format

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
        frame_detected = frame.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            class_id = int(labels[i])
            score = float(scores[i])
            
            # Detections bbox format for tracker
            if CLASS_NAMES[class_id] == "Person": # only track person
                per_detections.append([x1, y1, x2, y2, score])
                
            else:
                obj_detections.append([x1, y1, x2, y2, score, class_id])
            draw_bbox(frame_detected, CLASS_NAMES[class_id], x1, y1, x2, y2, score, type='detect')


        # Convert per_detections to numpy array
        if per_detections:
            per_detections = np.array(per_detections)
        else:
            per_detections = np.empty((0, 5))
        
        #! Update tracker with per_detections format
        online_targets = tracker.update(per_detections, [height, width], [height, width]) #img_info and img_size is for scaling img, if not then just pass [height, width]

        online_targets = detect_css_violations(online_targets, obj_detections) #! CSS violation
        # Draw tracked objects
        frame_tracked = frame.copy()
        frame_violated  = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            print("Missing:", t.missing)
            #* more conditions to filter out unwanted boxes
            # vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
            # if tlwh[2] * tlwh[3] > min_box_area and not vertical:

            if tlwh[2] * tlwh[3] > MIN_BOX_AREA:
                # # save results
                # tracking_results.append(
                #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                # )
                # Draw the bounding box
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, missing=t.missing, type='track')
                draw_bbox(frame_violated, tid, x1, y1, x2, y2, t.score, missing=t.missing, type='violate', class_names=CLASS_NAMES)

        # save and display the frame
        draw_fps(cap, frame_detected)
        draw_fps(cap, frame_tracked)
        draw_fps(cap, frame_violated)
        out_detect.write(frame_detected)
        out_track.write(frame_tracked)
        out_violate.write(frame_violated)
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out_detect.release()
    out_track.release()
    cv2.destroyAllWindows()
    print(f"Tracking results are saved in {OUTPUT_PATH}out_track_{num}_yolo.mp4")

if __name__ == "__main__":
    main()
