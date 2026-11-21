# DeepShield AI: Deepfake Face Detection

Real vs fake face detection using deep learning, with a Streamlit web app.

**Live demo:** add your deployed app URL here

## Overview

DeepShield AI detects AI-generated faces in images. It finds the face with MTCNN, classifies it with a fine-tuned ResNet-18, and returns a Real or Fake label with a confidence score and the inference time. It runs on CPU, so it can be deployed without a GPU.

## How it works

```
Image (JPG/PNG)
      ↓
MTCNN face detection and crop
      ↓
Resize to 256, center crop 224, ImageNet normalization
      ↓
ResNet-18 (single-output head) → sigmoid
      ↓
Label (Real / Fake) + confidence + inference time
```

Label convention: 0 = Fake, 1 = Real.

## Tech stack

| Component | Technology |
| --- | --- |
| Face detection | MTCNN (facenet-pytorch) |
| Classifier | ResNet-18, transfer learning |
| Framework | PyTorch, torchvision |
| Web app | Streamlit |

## Dataset

[Deepfake Face Images (Kaggle)](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images): real and AI-generated face images.

## Repository structure

```
deepfake-human-face/
├── app.py                          # Streamlit app (inference)
├── main.py                         # Entry script
├── deepfake_resnet.pth             # Trained model weights
├── Deepfake_image_detection.ipynb  # Experiments and evaluation
├── model.ipynb                     # Model training
├── requirements.txt                # Dependencies
├── pyproject.toml / uv.lock        # Project metadata
└── README.md
```

## Run locally

```bash
git clone https://github.com/TechwithKrishna/deepfake-human-face.git
cd deepfake-human-face
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501, upload a clear face photo, and click **Analyze**.

## Deploy

1. Push this repo to GitHub with `app.py`, `requirements.txt` and `deepfake_resnet.pth` in the root.
2. On Streamlit Community Cloud or Hugging Face Spaces (Streamlit SDK), set the main file to `app.py`.
3. Use Python 3.11 if the platform lets you choose.

## Limitations

- Works on images only (no video yet).
- Needs a detectable face, and works best on clear, front-facing photos.
- Performance can drop on low-quality images, extreme angles, occlusion, or faces unlike the training data.
- Confidence scores show model certainty, not proof. Do not use the output as the sole basis for decisions.

## Future work

- Video detection with frame-by-frame analysis
- Grad-CAM explainability
- Model ensembles and stronger backbones
- REST API with FastAPI

## References

- [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks (MTCNN)](https://arxiv.org/abs/1604.02878)
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385)

## License and disclaimer

For educational and research purposes only. Users are responsible for legal and ethical use.
