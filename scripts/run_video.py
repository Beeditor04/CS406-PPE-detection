import os
import subprocess

base_command = "python scripts/tracker_yolo.py --weights weights/best_yolo.pt --vid_dir sample/videos/{}.mp4"

for i in range(1, 4):
    command = base_command.format(i)
    print(f"Running: {command}")
    subprocess.run(command, shell=True)