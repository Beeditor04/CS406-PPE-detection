import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import numpy as np
import tempfile
import os
from io import BytesIO

# Set up page
st.set_page_config(layout="wide", page_title="PPE Detection Web")

# Apply white background
st.markdown(
    """
    <style>
        body {
            background-color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load YOLO model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

# Predict function
def predict_image(model, image):
    results = model(image)  # Dự đoán
    return results

# Draw bounding boxes
def draw_boxes(image, results, model, threshold=0.5):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for r in results[0].boxes.data.tolist():  # Lấy bounding boxes
        x1, y1, x2, y2, score, class_id = r
        if score >= threshold:
            class_name = model.names[int(class_id)]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1), f"{class_name}: {score:.2f}", fill="white")
    return img

def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return byte_im

# Main app
def main():
    st.title("PPE Detection Web")

    model = load_model("best.pt")

    # Tải lên file ảnh hoặc video
    uploaded_file = st.file_uploader(label = "Upload Image or Video", type = ["jpg", "jpeg", "png", "mp4"])
    if uploaded_file:
        file_type = uploaded_file.type.split('/')[0]

        # Xử lý ảnh
        if file_type == "image":
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            st.write("Running prediction...")
            results = predict_image(model, image)
            drawn_image = draw_boxes(image, results, model)

            st.image(drawn_image, caption="Predicted Image", use_container_width=True)
            st.write("### Download Processed Image")
            st.download_button("Download PPE Detection Image", convert_image(drawn_image), "ppe_detection.png", "image/png")


        # Xử lý video
        elif file_type == "video":
            # Lưu video gốc vào tệp tạm thời
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_input:
                temp_input.write(uploaded_file.read())  # Đọc tệp từ UploadedFile vào bộ nhớ tạm
                video_path = temp_input.name  # Lưu tệp tạm thời

            st.video(video_path)
            
            # Chạy dự đoán trên video
            cap = cv2.VideoCapture(video_path)
            output_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                predictions = predict_image(model, image)

                frame_with_boxes = np.array(draw_boxes(image, predictions, model))
                output_frames.append(cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR))

            cap.release()

            # Lưu video predict
            output_video_path = os.path.join(os.getcwd(), "output_video.mp4")
            height, width, _ = output_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (width, height))

            for frame in output_frames:
                out.write(frame)
            out.release()

            st.success("Video processing completed!")
            st.write("### Download Processed Video")
            st.download_button("Download Processed Video", open(output_video_path, "rb").read(), "output_video.mp4", "video/mp4")
            # Xoá các tệp tạm thời sau khi hiển thị
            os.remove(video_path)


if __name__ == "__main__":
    main()
