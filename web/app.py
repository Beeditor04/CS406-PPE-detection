import sys
import os

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tempfile
from io import BytesIO
from scripts.detect_yolo import inference as inference_yolo
from scripts.detect_faster_rcnn import inference as inference_faster_rcnn
from scripts.tracker_yolo import tracking
# Set up page
st.set_page_config(layout="wide", page_title="PPE Detection Web")

# Apply white background
st.markdown(
    """
    <style>
    .stImage {
        display: block;
        margin: 0 auto;
        max-width: 80%;
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }
    .caption {
        text-align: center;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Predict function
def predict_image(model, image):
    results = model(image)  # Dự đoán
    return results

def get_class_color(class_id):
    colors = [
        (128, 0, 0),    # Maroon
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (192, 192, 192),# Silver
        (128, 128, 128),# Gray
        (0, 0, 128),    # Navy
        (255, 255, 0),  # Yellow
        (0, 128, 0),    # Dark Green
        (128, 0, 128),  # Purple
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (255, 192, 203),# Pink
        (255, 222, 173),# Navajo White
        (173, 216, 230),# Light Blue
        (240, 230, 140),# Khaki
    ]
    return colors[class_id % len(colors)]

# # Draw bounding boxes
# def draw_boxes(image, results, model, threshold=0.5):
#     img = image.copy()
#     draw = ImageDraw.Draw(img)

#     for r in results[0].boxes.data.tolist():  # Lấy bounding boxes
#         x1, y1, x2, y2, score, class_id = r
#         if score >= threshold:
#             class_name = model.names[int(class_id)]
#             color = get_class_color(int(class_id))
#             draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
#             draw.text((x1, y1), f"{class_name}: {score:.2f}", fill=color)
#     return img

# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()
#     return byte_im

def uploaded_file_2_cv_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed.")
    return image

def cv2_image_to_bytes(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv2_img) # to pil
    buf = BytesIO() # back to bytes
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


# Main app
def main():
    st.title("PPE Detection Web")
    CLASS_NAMES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Gloves", "NO-Gloves", "Person"]

    # model = load_model("pretrain_new_data.pt")
    model_yolo = "weights/best_yolo.pt"
    model_frcnn = "weights/best_faster_rcnn.pt"
    # Tải lên file ảnh hoặc video
    uploaded_file = st.file_uploader(label = "Upload Image or Video", type = ["jpg", "jpeg", "png", "mp4"])
    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]

        # Xử lý ảnh
        if file_type == "image":
            #! process image
            # image = Image.open(uploaded_file).convert("RGB")
            # st.image(image, caption="Uploaded Image", use_container_width=True)

            # st.write("Running prediction...")
            # results = predict_image(model, image)
            # drawn_image = draw_boxes(image, results, model)

            # st.image(drawn_image, caption="Predicted Image", use_container_width=True)
            # st.write("### Download Processed Image")
            image = uploaded_file_2_cv_image(uploaded_file)
            # detection_result, violation_result = inference_yolo(weights=model_yolo, img_path=image, class_names=CLASS_NAMES, detect_thresh=0.5)
            detection_result, violation_result = inference_faster_rcnn(weights=model_frcnn, img_path=image, class_names=CLASS_NAMES, detect_thresh=0.5)
            #! show image results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection Result")
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(detection_result, caption="Detection Result", width=400)  
                    st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.subheader("Violation Detection Result")
                with st.container():
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(violation_result, caption="Violation Detection", width=400) 
                    st.markdown('</div>', unsafe_allow_html=True)

            #! Download processed image
            st.download_button("Download Detection", cv2_image_to_bytes(detection_result), "ppe_detection.png", "image/png")
            st.download_button("Download Violation Detection Image", cv2_image_to_bytes(violation_result), "violation_detection.png", "image/png")


        # Xử lý video
        elif file_type == "video":
            # Lưu video gốc vào tệp tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                temp_input.write(uploaded_file.read())  # Đọc tệp từ UploadedFile vào bộ nhớ tạm
                video_path = temp_input.name  # Lưu tệp tạm thời

            # st.video(video_path)
            
            # # Chạy dự đoán trên video
            # cap = cv2.VideoCapture(video_path)
            # output_frames = []

            # while cap.isOpened():
            #     ret, frame = cap.read()
            #     if not ret:
            #         break

            #     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #     predictions = predict_image(model, image)

            #     frame_with_boxes = np.array(draw_boxes(image, predictions, model))
            #     output_frames.append(cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR))
            # # Lưu video predict
            # output_video_path = os.path.join(os.getcwd(), "output_video.mp4")
            # height, width, _ = output_frames[0].shape
            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))
            # cap.release()
            # for frame in output_frames:
            #     out.write(frame)
            # out.release()
            st.write("Running prediction...")
            detect_path, track_path, violate_path = tracking(
                weights=model_yolo,
                video_path=video_path,
                class_names=CLASS_NAMES,
                detect_thresh=0.5
            )       

            # Display videos
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Detection")
                st.video(detect_path)
            with col2:
                st.subheader("Violation")
                st.video(violate_path)
            st.success("Video processing completed!")
            #! show video results

            # Hiển thị video đã xử lý
            st.write("### Download Processed Video")
                # Download buttons
            st.download_button("Download Detection", open(detect_path, "rb").read(), "detection.mp4", "video/mp4")
            st.download_button("Download Violation", open(violate_path, "rb").read(), "violation.mp4", "video/mp4")
            # st.download_button("Download Processed Video", open(output_video_path, "rb").read(), "output_video.mp4", "video/mp4")
            # Xoá các tệp tạm thời sau khi hiển thị
            os.remove(video_path)
            os.remove(detect_path)
            os.remove(track_path)
            os.remove(violate_path)


if __name__ == "__main__":
    main()
