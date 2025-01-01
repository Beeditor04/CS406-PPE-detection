import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from parsers.parser_tracker import parse_args
from utils.yaml_helper import read_yaml
from trackers.byte_tracker import BYTETracker


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
num_track = 1
num_detect = 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_track_{num_track}.mp4")):
    num_track += 1
while os.path.exists(os.path.join(OUTPUT_PATH, f"out_detect_{num_detect}.mp4")):
    num_detect += 1
out_track = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_track_{num_track}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))
out_detect = cv2.VideoWriter(os.path.join(OUTPUT_PATH, f"out_detect_{num_detect}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), int(cap.get(cv2.CAP_PROP_FPS)), (width, height))

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

def draw_fps(frame):
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = ' FPS: ' + str(fps) + ' Width: ' + str(cap.get(3)) + ' Height: ' + str(cap.get(4))
    cv2.putText(frame, fps, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def draw_bbox(frame, id, x1, y1, x2, y2, conf, type='detect'):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if type == "detect":
        cv2.putText(frame, f"{CLASS_NAMES[id]}: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if type == "track":
        cv2.putText(frame, f'ID: {id}, Score: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
    tracking_results = [] # store tracking results for eval, debug,...

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #! Detect
        # detect_results = model.detect(frame) # return xyxy format
        detect_results = model(frame) # return xyxy format

        detections = []
        frame_detected = frame.copy()
        for result in detect_results:
            for boxes in result.boxes:
                label, conf, bbox = int(boxes.cls[0]), float(boxes.conf[0]), boxes.xyxy.tolist()[0]
                # print(label, conf, bbox)
                x1, y1, x2, y2 = map(int, bbox) 
                class_id = int(label)
                
                #detections bbox format for tracker
                detections.append([x1, y1, x2, y2, conf]) 

                draw_bbox(frame_detected, class_id, x1, y1, x2, y2, conf, type='detect')

        # Convert detections to numpy array
        if detections:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 5))
        
        #! Update tracker with detections format
        online_targets = tracker.update(detections, [height, width], [height, width]) #img_info and img_size is for scaling img, if not then just pass [height, width]

        # Draw tracked objects
        frame_tracked = frame.copy()
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id

            #* more conditions to filter out unwanted boxes
            # vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
            # if tlwh[2] * tlwh[3] > min_box_area and not vertical:

            if tlwh[2] * tlwh[3] > MIN_BOX_AREA:
                # save results
                tracking_results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )
                # Draw the bounding box
                x1, y1, w, h = map(int, tlwh)
                x2, y2 = x1 + w, y1 + h
                draw_bbox(frame_tracked, tid, x1, y1, x2, y2, t.score, type='track')

        # save and display the frame
        draw_fps(frame_detected)
        draw_fps(frame_tracked)
        out_detect.write(frame_detected)
        out_track.write(frame_tracked)

        frame_id += 1
        # cv2.imshow('frame', frame_tracked)
        # cv2.imshow('frame', frame_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out_detect.release()
    out_track.release()
    cv2.destroyAllWindows()
    print(f"Tracking results are saved in {OUTPUT_PATH}out_track_{num_track}.mp4")

if __name__ == "__main__":
    main()
