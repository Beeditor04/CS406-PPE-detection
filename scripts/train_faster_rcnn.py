import sys
import os
import datetime
from tqdm import tqdm
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from torchvision import models 
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim import SGD
from torchvision import transforms
from torch.utils.data import DataLoader

from models.FASTER_RCNN import faster_rcnn
from loader.loader_faster_rcnn import CustomDataset
from utils.data_helper import collate_fn
from utils.detection.coco_eval import CocoEvaluator
from utils.detection.coco_utils import get_coco_api_from_dataset
from utils.detection.metric_logger import MetricLogger
from parsers.parser_faster_rcnn import parse_args


import time

def get_preprocessed_data(data_path):
    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    data = CustomDataset(
        root_dir=data_path,
        transform=transform
    )
    return data

def evaluate_model(model, val_loader, device):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Evaluation:"

    # Get COCO format dataset
    coco = get_coco_api_from_dataset(val_loader.dataset)
    iou_types = ['bbox']  # For Faster R-CNN
    coco_evaluator = CocoEvaluator(coco, iou_types)

    with torch.inference_mode():
        for images, targets in metric_logger.log_every(val_loader, 100, header):
            images = list(img.to(device) for img in images)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            model_time = time.time()
            outputs = model.model(images)  # Use model.model since we're using our wrapper class
            print(f"Targets: {targets}")
            print(f"Outputs: {outputs}")
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
            model_time = time.time() - model_time
            
            # Convert bounding boxes from [x_min, y_min, x_max, y_max] to [x_min, y_min, width, height]
            for output in outputs:
                boxes = output['boxes']
                # Calculate width and height
                widths = boxes[:, 2] - boxes[:, 0]
                heights = boxes[:, 3] - boxes[:, 1]
                # Stack [x_min, y_min, width, height] into a new tensor
                output['boxes'] = torch.stack([boxes[:, 0], boxes[:, 1], widths, heights], dim=1)

            # Create 'res' list in the specified format
            res = [{
                "image_id": target["image_id"].item(),
                "boxes": output["boxes"],
                "labels": output["labels"],
                "scores": output["scores"]
            } for target, output in zip(targets, outputs)]
            print(f"Outputs converted: {res}")
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    coco_evaluator.synchronize_between_processes()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    
    torch.set_num_threads(n_threads)
    return coco_evaluator

# def evaluate_model(model, val_loader, device):
#     model.eval()
#     metric = MeanAveragePrecision()
#     with torch.no_grad():
#         for images, targets in tqdm(val_loader, desc="Evaluating"):
#             images = [img.to(device) for img in images]
#             targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

#             preds = model.forward(images)

#             pred_dicts = []
#             for p in preds:
#                 pred_dicts.append({
#                     "boxes": p["boxes"].detach().cpu(),
#                     "scores": p["scores"].detach().cpu(),
#                     "labels": p["labels"].detach().cpu(),
#                 })

#             target_dicts = []
#             for t in targets:
#                 target_dicts.append({
#                     "boxes": t["boxes"].detach().cpu(),
#                     "labels": t["labels"].detach().cpu(),
#                 })

#             metric.update(pred_dicts, target_dicts)

#     results = metric.compute()
#     return results["map"]

def train_model(train_loader, val_loader, device, num_classes=5, epochs=10, batch_size=1, eval_every=1):
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    model = faster_rcnn(num_classes)
    model.model.to(device)
    for param in model.model.backbone.parameters():
        param.requires_grad = False
    params = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Training started.")
    print("Device: ", device)
    print("Number of classes: ", num_classes)
    print("Number of epochs: ", epochs)
    print("Batch size: ", batch_size)
    print("train_loader: ", len(train_loader))

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (images, targets) in enumerate(train_loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            with open(os.path.join(log_dir, "training.log"), "a") as log_file:
                log_file.write(f"Epoch: {epoch}, Iteration: {i}\n")
                log_file.write(f"Targets: {targets}\n")
            loss_dict = model.forward(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            running_loss += losses.item()
            print(f"Epoch [{epoch+1}/{epochs}], Iteration [{i}/{len(train_loader)}], Loss: {losses.item()}")
        lr_scheduler.step()
        # if eval_every > 0 and (epoch + 1) % eval_every == 0:
        #     mean_ap = evaluate_model(model, val_loader, device)
        #     print(f"[Epoch {epoch+1}] mAP: {mean_ap}")

    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.model.state_dict(), f"weights/faster-rcnn-{timestamp}.pt")
    print("Training complete.")

def main():
    args = parse_args()

    FOLDER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', args.data_dir))
    TRAIN_PATH = os.path.join(FOLDER_PATH, 'train')
    VAL_PATH = os.path.join(FOLDER_PATH, 'val')
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    EVAL_EVERY = args.eval_every

    # Create dataset and dataloader
    train_dataset = get_preprocessed_data(TRAIN_PATH)
    val_dataset = get_preprocessed_data(VAL_PATH)

    print("Number of training samples: ", len(train_dataset))
    img, target = train_dataset[0]
    print("Image Size: ", img.size())
    print("Target sample: ", target)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn,
        pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(
        train_loader,
        val_loader,
        device,
        num_classes=5,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        eval_every=EVAL_EVERY
    )

if __name__ == "__main__":
    main()