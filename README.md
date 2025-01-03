
# Before deploying web
```
git clone https://github.com/Beeditor04/CS406-PPE-detection.git
pip install -r requirements.txt
python setup.py develop
```

# Preparing 
you can download our `faster-rcnn` and `yolov5n` model here:
- `faster-rcnn`: [drive](https://drive.google.com/file/d/1ciFtmC6eh0wRoK3yU4rS1JlmdH_ee01i/view?usp=drive_link)
- `yolov5n`: [drive](https://drive.google.com/file/d/1VuorH4fgbafaroALJNafHeSMQdweAHsz/view?usp=drive_link)

And then put it in folder `weights`.

# Deploy web
```
streamlit run web/app.py
```

# Model
- train faster_rcnn (remember to create `.yaml` file, put it in the folder `data/` or wherever you want)
- `data-ppe.yaml`: [drive](https://drive.google.com/file/d/1P9rRNMd3ErvO47f3xOn6nnsla0waxRC7/view?usp=drive_link) 
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
- run tracker yolo
```
python scripts/tracker_yolo.py --weights weights/best_yolo.pt --vid_dir sample/videos/1.mp4
```
---
> *the code below is not yet finished*
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


# Tracking and PPE violation detection
- run tracker faster_rcnn
```
python scripts/tracker_faster_rcnn.py --weights weights/best_faster_rcnn.pt --vid_dir sample/videos/1.mp4
```


