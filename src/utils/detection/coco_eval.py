import numpy as np
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import contextlib
import os

class CocoEvaluator:
    def __init__(self, coco_gt, iou_types):
        self.coco_gt = coco_gt
        self.iou_types = iou_types
        self.coco_eval = {iou_type: COCOeval(coco_gt, iouType=iou_type) for iou_type in iou_types}
        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def update(self, predictions):
        img_ids = list(np.unique([pred["image_id"] for pred in predictions]))
        self.img_ids.extend(img_ids)

        print(f"Image IDs: {img_ids} with predictions: {predictions}")
        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            print(f"COCO converted Results: {results}")
            coco_dt = self.coco_gt.loadRes(results) if results else COCO()
            self.coco_eval[iou_type].cocoDt = coco_dt
            self.coco_eval[iou_type].params.imgIds = list(img_ids)
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                self.coco_eval[iou_type].evaluate()


    def synchronize_between_processes(self):
        for iou_type in self.iou_types:
            self.coco_eval[iou_type].accumulate()
            self.coco_eval[iou_type].summarize()

    def accumulate(self):
        for iou_type in self.iou_types:
            self.coco_eval[iou_type].accumulate()

    def summarize(self):
        for iou_type in self.iou_types:
            print(f"\nIOU metric: {iou_type}")
            self.coco_eval[iou_type].summarize()

    def prepare(self, predictions, iou_type):
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        else:
            raise ValueError(f"Unknown iou type {iou_type}")

    def prepare_for_coco_detection(self, predictions):
        coco_results = []
        for pred in predictions:
            boxes = pred["boxes"].tolist()
            scores = pred["scores"].tolist()
            labels = pred["labels"].tolist()
            image_id = pred["image_id"]

            for box, score, label in zip(boxes, scores, labels):
                # Skip background label=0
                if label == 0:
                    continue
                # Otherwise, use label as the category_id
                coco_results.append({
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": box,    # Already in [x_min, y_min, width, height]
                    "score": score,
                })
        return coco_results
    
    