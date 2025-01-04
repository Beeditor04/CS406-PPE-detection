import sys
import os
import cv2
import time
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from ultralytics import YOLO
from parsers.parser_tracker import parse_args
from utils.yaml_helper import read_yaml
from utils.function import non_max_suppression, draw_bbox, draw_fps
from utils.detect_css_violations import detect_css_violations
from trackers.byte_tracker import BYTETracker

class TrackerArgs:
    def __init__(self, track_thresh, track_buffer, match_thresh, fuse_score):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.fuse_score = fuse_score
        self.mot20 = False

def tracking(weights="weights/best_yolo.pt", video_path=None, class_names=None, detect_thresh=0.3, device="cpu"):
    # Setup params
    TRACK_THRESH = 0.5
    TRACK_BUFFER = 30
    MATCH_THRESH = 0.85
    FUSE_SCORE = False
    MIN_BOX_AREA = 100
    DETECT_THRESH = detect_thresh
    CLASS_NAMES = class_names

    # Initialize model and tracker
    model = YOLO(weights, detect_thresh)
    tracker_args = TrackerArgs(TRACK_THRESH, TRACK_BUFFER, MATCH_THRESH, FUSE_SCORE)
    tracker = BYTETracker(tracker_args)

    # Open video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Create output directory if it doesn't exist
    OUTPUT_PATH = "output/videos/"
    if not os.path.exists(OUTPUT_PATH):
        if not os.path.exists("output/"):
            os.makedirs("output/")
        os.makedirs("output/videos")

    # Create temporary output videos
    num = 1
    while os.path.exists(os.path.join(OUTPUT_PATH, f"out_track_{num}_yolo.mp4")):
        num += 1

    detect_path = os.path.join(OUTPUT_PATH, f"out_detect_{num}_yolo.mp4")
    track_path = os.path.join(OUTPUT_PATH, f"out_track_{num}_yolo.mp4")
    violate_path = os.path.join(OUTPUT_PATH, f"out_violate_{num}_yolo.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_detect = cv2.VideoWriter(detect_path, fourcc, fps, (width, height))
    out_track = cv2.VideoWriter(track_path, fourcc, fps, (width, height))
    out_violate = cv2.VideoWriter(violate_path, fourcc, fps, (width, height))


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        start_time = time.time()
        # Detect
        detect_results = model(frame)

        # Post-processing: filter low score, nms
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

        keep = non_max_suppression(boxes, scores, iou_threshold=0.5)
        boxes = [boxes[i] for i in keep]
        scores = [scores[i] for i in keep]
        labels = [labels[i] for i in keep]

        per_detections = []
        obj_detections = []
        frame_detected = frame.copy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            class_id = int(labels[i])
            score = float(scores[i])

            if CLASS_NAMES[class_id] == "Person":
                per_detections.append([x1, y1, x2, y2, score])
            else:
                obj_detections.append([x1, y1, x2, y2, score, class_id])
            draw_bbox(frame_detected, class_id, x1, y1, x2, y2, score, type='detect', class_names=CLASS_NAMES)

        if per_detections:
            per_detections = np.array(per_detections)
        else:
            per_detections = np.empty((0, 5))

        online_targets = tracker.update(per_detections, [height, width], [height, width])

        online_targets = detect_css_violations(online_targets, obj_detections)
        frame_tracked = frame.copy()
        frame_violated = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            if tlwh[2] * tlwh[3] > MIN_BOX_AREA:
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, missing=t.missing, type='track')
                draw_bbox(frame_violated, tid, x1, y1, x2, y2, t.score, missing=t.missing, type='violate', class_names=CLASS_NAMES)

        # Calculate FPS
        process_time = time.time() - start_time
        fps = 1 / process_time

        draw_fps(cap, frame_detected, fps)
        draw_fps(cap, frame_tracked, fps)
        draw_fps(cap, frame_violated, fps)
        out_detect.write(frame_detected)
        out_track.write(frame_tracked)
        out_violate.write(frame_violated)

    cap.release()
    out_detect.release()
    out_track.release()
    out_violate.release()
    print(f"Output videos saved at {detect_path}")
    return detect_path, track_path, violate_path

if __name__ == "__main__":
    # SETUP parse
    args = parse_args()

    # yaml class
    yaml_class = read_yaml(args.yaml_class)
    CLASS_NAMES = yaml_class["names"]

    # Call tracking function
    tracking(
        weights=args.weights,
        video_path=args.vid_dir,
        class_names=CLASS_NAMES,
        detect_thresh=args.detect_thresh,
        device=args.device
    )