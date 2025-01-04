import os 
import argparse

# Arg for main path
parser = argparse.ArgumentParser()
parser.add_argument("--main", type=str, default="data/split")
args = parser.parse_args()

LABEL_PATH = "labels/"
IMG_PATH = "images/"
main = args.main
folders = ["train", "val", "test"]

# Stats
print("============")
total_labels = 0
total_images = 0
for f in folders:
    print("Folder:", f)
    total_images += len(os.listdir(os.path.join(main, f, IMG_PATH)))
    total_labels += len(os.listdir(os.path.join(main, f, LABEL_PATH)))
    print("Total labels:", len(os.listdir(os.path.join(main, f, LABEL_PATH))))
    print("Total images:", len(os.listdir(os.path.join(main, f, IMG_PATH))))
print("Overall labels:", total_labels)
print("Overall images:", total_images)