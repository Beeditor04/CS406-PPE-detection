
# Dataset
- split dataset
```
python scripts/split_dataset.py --data_dir data/ --output_dir data/split/ --train_size 0.7 --test_size 0.1
```
- split dataset 0.1
```
python scripts/split_dataset_01.py --data_dir data/ --output_dir data/split_01/ --train_size 0.7 --test_size 0.1
```
- count dataset
```
python scripts/count.py --main data/split
```

- valid dataset
```
python scripts/is_valid_dataset.py

```

# Model
- train faster_rcnn (remember to create `.yaml` file, put it in the folder)
```
python scripts/train_faster_rcnn.py --data_dir "data/split" --batch_size 8 --epochs 10 --eval_every 5 --iter_every 5 --num_classes 7 --yaml "data/data-ppe.yaml" --lr 0.005 --resize 640 --is_aug 0
```

- run img faster_rcnn inference
```
python scripts/detect_faster_rcnn.py --weights weights/best_faster_rcnn.pt --img_path sample/images/1.jpg
```

- run img yolo inference
```
python scripts/detect_yolo.py --weights weights/best_yolo.pt --img_path sample/images/1.jpg
```
# Tracking
- run tracker yolo
```
python scripts/tracker_yolo.py --weights weights/best_yolo.pt --vid_dir sample/videos/1.mp4
```

- run tracker faster_rcnn
```
python scripts/tracker_faster_rcnn.py --weights weights/best_faster_rcnn.pt --vid_dir sample/videos/1.mp4
```
# PPE violation


