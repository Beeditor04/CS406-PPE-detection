import cv2
import os
import numpy as np

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
    
    