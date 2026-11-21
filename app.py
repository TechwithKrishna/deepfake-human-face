import time
from pathlib import Path

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image, ImageOps

# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Deepfake Detection System", page_icon="🕵️")

# -----------------------
# Device and paths
# -----------------------
DEVICE = torch.device("cpu")
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "deepfake_resnet.pth"


# -----------------------
# MTCNN face detector
# -----------------------
@st.cache_resource
def get_mtcnn():
    return MTCNN(keep_all=False, device=DEVICE)


# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Make sure 'deepfake_resnet.pth' is in the same folder as app.py."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    # Support checkpoints saved as {"state_dict": ...} or {"model_state_dict": ...}
    state_dict = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break

    # Remove "module." prefix if the model was saved from DataParallel
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model = models.resnet18(weights=None)
    model.fc = build_head(state_dict)

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def build_head(state_dict):
    """Rebuild the final classifier layer(s) to match the saved weights."""
    # Single linear layer: fc.weight / fc.bias
    if "fc.weight" in state_dict:
        out_f, in_f = state_dict["fc.weight"].shape
        return torch.nn.Linear(in_f, out_f)

    # Sequential head: fc.0.weight, fc.3.weight, fc.6.weight, ...
    linear_idx = sorted(
        {
            int(k.split(".")[1])
            for k in state_dict
            if k.startswith("fc.") and k.endswith(".weight")
        }
    )
    if not linear_idx:
        raise ValueError("Could not find classifier weights (fc.*) in the model file.")

    layers = []
    for pos in range(linear_idx[-1] + 1):
        if pos in linear_idx:
            out_f, in_f = state_dict[f"fc.{pos}.weight"].shape
            layers.append(torch.nn.Linear(in_f, out_f))
        elif layers and isinstance(layers[-1], torch.nn.Linear):
            layers.append(torch.nn.ReLU())
        else:
            layers.append(torch.nn.Dropout(0.3))
    return torch.nn.Sequential(*layers)


# -----------------------
# Image transform (ResNet standard)
# -----------------------
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


# -----------------------
# Face detection
# -----------------------
def detect_face(image: Image.Image, mtcnn):
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None

    width, height = image.size
    x1, y1, x2, y2 = boxes[0]

    # Keep the crop box inside the image bounds
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width, int(x2))
    y2 = min(height, int(y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


# -----------------------
# Prediction
# 0 = Fake, 1 = Real
# -----------------------
def predict(image: Image.Image, model, mtcnn):
    start = time.perf_counter()

    face = detect_face(image, mtcnn)
    if face is None:
        return None, 0.0, time.perf_counter() - start

    tensor = transform(face).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor)
        if output.numel() == 1:
            # Single logit: sigmoid gives P(Real)
            prob_real = torch.sigmoid(output).item()
        else:
            # Two logits: index 0 = Fake, index 1 = Real
            prob_real = torch.softmax(output, dim=1)[0, 1].item()

    label = "Real" if prob_real >= 0.5 else "Fake"
    confidence = prob_real if label == "Real" else 1 - prob_real
    elapsed = time.perf_counter() - start
    return label, confidence, elapsed


# -----------------------
# UI
# -----------------------
st.title("🕵️ Deepfake Detection System")
st.caption("Upload a clear, front-facing photo of a face to check whether it looks real or AI-generated.")

try:
    model = load_model()
    mtcnn = get_mtcnn()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")
    except Exception:
        st.error("Could not read this image. Please upload a valid JPG or PNG file.")
        st.stop()

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            label, confidence, elapsed = predict(image, model, mtcnn)

        if label is None:
            st.warning("No face detected. Please upload a clear, front-facing photo.")
        else:
            st.subheader(f"Result: {label}")
            st.write(f"Confidence: {confidence:.2%}")
            st.progress(min(max(confidence, 0.0), 1.0))
            st.caption(f"Inference time: {elapsed * 1000:.0f} ms")

            if label == "Fake":
                st.error("⚠️ This image is likely a deepfake.")
            else:
                st.success("✅ This image appears real.")

st.markdown("---")
st.caption(
    "This is a research demo. Results are not definitive proof and can be wrong, "
    "especially for low-quality images, extreme angles, or faces outside the training data."
)
