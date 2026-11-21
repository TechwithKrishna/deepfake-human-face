import time
from pathlib import Path

import streamlit as st
import torch
from PIL import Image, ImageOps
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
import torchvision.models as models

st.set_page_config(page_title="Deepfake Detection System", page_icon="🕵️")

# -----------------------
# Device and model path
# -----------------------
device = torch.device("cpu")
MODEL_PATH = Path(__file__).resolve().parent / "deepfake_resnet.pth"

# -----------------------
# MTCNN Face Detector
# -----------------------
@st.cache_resource
def get_mtcnn():
    return MTCNN(keep_all=False, device=device)

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Keep 'deepfake_resnet.pth' in the same folder as app.py."
        )
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

try:
    mtcnn = get_mtcnn()
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# -----------------------
# Image Transform (ResNet standard)
# -----------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------
# Face Detection
# -----------------------
def detect_face(image: Image.Image):
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None

    w, h = image.size
    x1, y1, x2, y2 = boxes[0]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))

# -----------------------
# Prediction (0 = Fake, 1 = Real)
# -----------------------
def predict(image: Image.Image):
    start = time.perf_counter()
    face = detect_face(image)
    if face is None:
        return None, 0.0, time.perf_counter() - start

    tensor = transform(face).unsqueeze(0).to(device)
    with torch.no_grad():
        prob_real = torch.sigmoid(model(tensor)).item()

    label = "Real" if prob_real >= 0.5 else "Fake"
    confidence = prob_real if label == "Real" else 1 - prob_real
    return label, confidence, time.perf_counter() - start

# -----------------------
# UI
# -----------------------
st.title("🕵️ Deepfake Detection System")
st.caption("Upload a clear, front-facing photo of a face.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    except Exception:
        st.error("Could not read this image. Please upload a valid JPG or PNG.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            label, confidence, elapsed = predict(image)

        if label is None:
            st.warning("No face detected. Please upload a clear, front-facing photo.")
        else:
            st.subheader(f"Result: {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.caption(f"Inference time: {elapsed * 1000:.0f} ms")
            if label == "Fake":
                st.error("⚠️ This image is likely a deepfake.")
            else:
                st.success("✅ This image appears real.")
