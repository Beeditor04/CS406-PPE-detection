import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision import models 
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        # Load corresponding label
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)
        
        # Read YOLO format labels (class x_center y_center width height)
        boxes = []
        classes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    classes.append(int(class_id) + 1) #! Increment class by 1 for COCO
        with open('debug-before.log', 'a') as f:
            f.write(f"BBOX before transform: {boxes} with {image.size}\n")

        boxes = torch.tensor(boxes, dtype=torch.float16)
        classes = torch.tensor(classes, dtype=torch.int64)
        
        if self.transform:
            image = self.transform(image)
            if boxes.numel() > 0:
                boxes[:, 0] = boxes[:, 0] * original_size[0]
                boxes[:, 1] = boxes[:, 1] * original_size[1]
                boxes[:, 2] = boxes[:, 2] * original_size[0]
                boxes[:, 3] = boxes[:, 3] * original_size[1]

                x_center = boxes[:, 0]
                y_center = boxes[:, 1]
                width = boxes[:, 2]
                height = boxes[:, 3]
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                boxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)

                # Resize bounding boxes to match the resized image
                new_size = image.size()[1:]  # (C, H, W) -> (H, W)
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * (new_size[1] / original_size[0])
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * (new_size[0] / original_size[1]) 
        target = {'boxes': boxes, 'labels': classes, 'image_id': torch.tensor([idx])}
        return image, target
    
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