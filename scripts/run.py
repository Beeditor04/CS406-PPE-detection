import os
import subprocess

# Define the base command
base_command = "python scripts/detect_faster_rcnn.py --weights weights/best_faster_rcnn.pt --img_path sample/images/{}.jpg"

# Loop to run the command 12 times with different values of X
for i in range(1, 13):
    command = base_command.format(i)
    print(f"Running: {command}")
    subprocess.run(command, shell=True)