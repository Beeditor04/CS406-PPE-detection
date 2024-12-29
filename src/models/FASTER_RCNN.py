import torch
import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.detection.coco_eval import CocoEvaluator
from utils.detection.coco_utils import get_coco_api_from_dataset
from utils.detection.metric_logger import MetricLogger

class faster_rcnn:
    def __init__(self, num_classes):
        self.weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=self.weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)

    def eval(self):
        self.train(False)

    def evaluate(self, val_loader, device):
        self.eval()
        coco = get_coco_api_from_dataset(val_loader.dataset)
        iou_types = ["bbox"]
        coco_evaluator = CocoEvaluator(coco, iou_types)
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                
                outputs = self.model(images)
                res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
                coco_evaluator.update(res)
        
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        metrics = {
            'mAP@0.5': coco_evaluator.coco_eval['bbox'].stats[1],
            'mAP@0.75': coco_evaluator.coco_eval['bbox'].stats[2],
            'mAP@0.5:0.95': coco_evaluator.coco_eval['bbox'].stats[0]
        }
        
        return metrics
