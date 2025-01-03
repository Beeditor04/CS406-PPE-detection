from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
def compute_metrics(all_preds, all_targets, class_names):
    # Configure mAP
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        iou_thresholds=[0.5],
        class_metrics=True
    )

    # all_preds / all_targets are lists of list-of-dicts -> extract boxes and labels
    for batch_preds, batch_targets in tqdm(zip(all_preds, all_targets), total=len(all_preds)):
        metric.update(batch_preds, batch_targets)

    results = metric.compute()
    # Per-class AP and AR at IoU=0.5
    ap_per_class = results["map_per_class"]
    ar_per_class = results["mar_100_per_class"]
    
    # mAP@0.5, mAR@0.5
    map50 = results["map"].item()          # same as "map_50" since we only have 0.5 in iou_thresholds
    mar50 = results["mar_100"].item()      # recall at maxDets=100 for IoU=0.5
    return {
        "map50": map50,
        "mar50": mar50,
        "ap50_per_class": ap_per_class,
        "ar50_per_class": ar_per_class
    }

def associate_score(person_box, obj_box):
    # Find intersection
    x1 = max(person_box[0], obj_box[0])
    y1 = max(person_box[1], obj_box[1])
    x2 = min(person_box[2], obj_box[2])
    y2 = min(person_box[3], obj_box[3])
    
    # Calculate areas
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    obj_area = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])
    
    # Return % of obj_box that intersects with person_box
    return intersection / obj_area if obj_area > 0 else 0