import torch
import torchvision
from torchvision import models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class faster_rcnn:
    def __init__(self, num_classes):
        self.weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=self.weights)
        self.num_classes = num_classes + 1  # +1 for background class
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def forward(self, images, targets=None):
        if targets is not None:
            # Validate class labels
            for target in targets:
                labels = target['labels']
                if torch.any(labels >= self.num_classes + 1) or torch.any(labels < 0):
                    print(f"Invalid labels found: {labels}")
                    print(f"Max label: {labels.max()}, Min label: {labels.min()}")
                    raise ValueError(f"Labels must be in range [1, {self.num_classes}]")
        if self.training:
            return self.model(images, targets)
        else:
            return self.model(images)

    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)

    def eval(self):
        self.train(False)

