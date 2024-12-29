import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


import torch
from PIL import Image
from models.FASTER_RCNN import faster_rcnn
from utils.function import show_preds
from torchvision import models, transforms

def inference(device, pretrain="weights/faster-rcnn.pt", src_path="sample/1.jpg"):
    model = faster_rcnn(5)
    model.model.load_state_dict(torch.load(pretrain))
    model.model.to(device)
    model.eval()

    threshold = 0.2

    weights = models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    normalize = weights.transforms()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    with torch.no_grad():
        image = Image.open(src_path).convert('RGB')
        print(image.size)
        image_tensor = transform(image).unsqueeze(0).to(device)
        image = image.resize((224, 224))
        preds = model.model(image_tensor)
        preds = [{k: v.to(device) for k, v in t.items()} for t in preds]
        print(preds)

    show_preds(image, image_tensor, preds, threshold=threshold)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inference(device, pretrain="weights/faster-rcnn-20241229-223244.pt", src_path="sample/1.jpg")