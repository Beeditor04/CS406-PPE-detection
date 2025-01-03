import argparse

def create_parser():
    parser = argparse.ArgumentParser(description="Parser for Tracker training")
    parser.add_argument("--vid_dir", type=str, default="sample/videos/1.mp4", help="Path to video directory")
    parser.add_argument("--yaml_class", type=str, default="data/data-ppe.yaml", help="Path to yaml file")
    parser.add_argument("--weights", type=str, default="weights/best_yolo.pt", help="Path to weights file")
    parser.add_argument("--detect_thresh", type=float, default=0.3, help="Detection threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    return parser

def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
