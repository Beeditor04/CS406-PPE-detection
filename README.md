- split dataset
```
python scripts/split_dataset.py --data_dir data/ --output_dir data/split/ --train_size 0.7 --test_size 0.1
```
- split dataset 0.1
```
python scripts/split_dataset_01.py --data_dir data/ --output_dir data/split_01/ --train_size 0.7 --test_size 0.1
```

- train faster_rcnn (remember to create `.yaml` file, put it in the folder)
```
python scripts/train_faster_rcnn.py --data_dir "data/split" --batch_size 8 --epochs 10 --eval_every 5 --iter_every 5 --num_classes 7 --yaml "data/data-ppe.yaml" --lr 0.005
```

- run inference
```
python scripts/inference.py
```

- count dataset
```
python scripts/count.py --main data/split
```

- valid dataset
```
python scripts/is_valid_dataset.py
```
