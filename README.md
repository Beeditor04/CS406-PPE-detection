- split dataset
```
python scripts/split_dataset.py --data_dir data/ --output_dir data/split/ --train_size 0.7 --test_size 0.1
```

- train faster_rcnn
```
python scripts/train_faster_rcnn.py --data_dir data/split --batch_size 8 --epochs 10 --eval_every 5
```