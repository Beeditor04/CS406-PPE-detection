import cv2
import os
import numpy as np
import torch

def show_preds(image, image_tensor, preds, output_dir="output", threshold=0.5):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find the next available file number
    num = 1
    while os.path.exists(os.path.join(output_dir, f"inference-{num}.jpg")):
        num += 1

    boxes = preds[0]['boxes']
    labels = preds[0]['labels']
    scores = preds[0]['scores']
    keep = non_max_suppression(boxes, scores, iou_threshold=0.1)
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    image_tensor = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    for i in range(boxes.shape[0]):
        if scores[i] < threshold:
            continue
        box = boxes[i].cpu().numpy()
        label = labels[i].cpu().numpy()
        score = scores[i].cpu().numpy()
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            (0, 255, 0), 2
        )
        cv2.putText(
            image,
            f"{label} {score:.2f}",
            (int(box[0]), int(box[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )
        # Save the image
    output_path = os.path.join(output_dir, f"inference-{num}.jpg")
    cv2.imwrite(output_path, image)
    print(f"Saved inference result to {output_path}")
    


def non_max_suppression(boxes, scores, iou_threshold):
    # Input validation
    if len(boxes) == 0 or len(scores) == 0:
        return []
    
    # Convert to tensor if not already
    if not isinstance(boxes, torch.Tensor):
        boxes = torch.tensor(boxes)
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores)
    
    # Ensure boxes and scores have same first dimension
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes and scores must have same length, got {boxes.shape[0]} and {scores.shape[0]}")

    # Handle single box case
    if boxes.shape[0] == 1:
        return [0]

    # Get coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Compute areas
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    # Sort by score
    _, order = scores.sort(0, descending=True)
    order = order.reshape(-1)  # Ensure order is 1D

    keep = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order[0].item())
            break
            
        i = order[0].item()
        keep.append(i)

        # Compute IoU
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.tensor(0.0), xx2 - xx1 + 1)
        h = torch.max(torch.tensor(0.0), yy2 - yy1 + 1)

        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # Keep boxes with IoU less than threshold
        mask = iou <= iou_threshold
        if not mask.any():
            break
            
        inds = mask.nonzero().reshape(-1)
        order = order[inds + 1]

    return keep