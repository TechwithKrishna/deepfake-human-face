import streamlit as st
import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms
import torchvision.models as models

# -----------------------
# Device
# -----------------------
device = torch.device("cpu")

# -----------------------
# MTCNN Face Detector
# -----------------------
@st.cache_resource
def get_mtcnn():
    return MTCNN(keep_all=False, device=device)

mtcnn = get_mtcnn()

# -----------------------
# Load Model
# -----------------------
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(
        torch.load(r"C:\Users\MAHESH KUMAR\Desktop\Intern projects\Deepfake model\Model\deepfake_resnet.pth", map_location=device)
    )

    model.to(device)
    model.eval()
    return model

model = load_model()

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

    if boxes is None:
        return None

    x1, y1, x2, y2 = map(int, boxes[0])
    return image.crop((x1, y1, x2, y2))

# -----------------------
# Prediction
# 0 = Fake, 1 = Real
# -----------------------
def predict(image: Image.Image):
    face = detect_face(image)

    if face is None:
        return "No face detected", 0.0

    tensor = transform(face).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob_real = torch.sigmoid(output).item()

    label = "Real" if prob_real >= 0.5 else "Fake"
    confidence = prob_real if label == "Real" else 1 - prob_real

    return label, confidence

# -----------------------
# UI
# -----------------------
st.title("🕵️ Deepfake Detection System")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        label, confidence = predict(image)

        st.subheader(f"Result: {label}")
        st.write(f"Confidence: {confidence:.2f}")

        if label == "Fake":
            st.error("⚠️ This image is likely a deepfake.")
        elif label == "Real":
            st.success("✅ This image appears real.")
        else:
            st.warning("No face detected.")