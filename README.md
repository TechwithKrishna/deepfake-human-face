🕵️‍♂️ DeepShield AI
Real vs Fake Face Detection using Deep Learning 🔥

“If you can’t trust what you see, what can you trust?”

Deepfakes are evolving fast.
So I built DeepShield AI — a lightweight deep learning system that detects whether a face is real or AI-generated in seconds.

🚀 Live System Overview

Upload an image → AI analyzes the face → Get instant prediction:

🟢 REAL (Authentic Human Face)
🔴 FAKE (AI-Generated / Deepfake)
📊 Confidence Score
⚡ Why this project matters

Deepfakes are no longer research problems — they are:

🎭 Identity fraud risks
📰 Misinformation tools
🔐 Security threats

DeepShield AI is a step toward restoring visual trust.

🧠 How it works (Simple but powerful)
Image Upload
   ↓
MTCNN Face Detection 👤
   ↓
Face Cropping & Cleaning ✂️
   ↓
ResNet Deep Learning Model 🧠
   ↓
Prediction (Real / Fake)
   ↓
Confidence Score 📊
🏗️ Tech Stack
🐍 Python
🔥 PyTorch (Deep Learning)
🧠 ResNet18 (Transfer Learning)
👤 MTCNN (Face Detection)
🌐 Streamlit (Web App)
📦 NumPy, Pillow
📊 Dataset Used

This project uses the Deepfake Face Images Dataset from Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images

Includes:

Real human face images
AI-generated / manipulated face images
Balanced binary classification setup
🧠 Model Details
Backbone: ResNet18
Training Type: Transfer Learning
Output:
0 → Fake
1 → Real
Activation: Sigmoid
Input Size: 224 x 224
📁 Project Structure
deepfake-detection/
│
├── app/                 # Streamlit frontend
│   └── app.py
│
├── src/                 # Core ML pipeline
│   ├── model.py
│   ├── predict.py
│   ├── preprocessing.py
│
├── models/             # Trained weights (.pth)
├── utils/
├── notebooks/          # Experiments & training
├── requirements.txt
└── README.md
🖥️ Run Locally
1️⃣ Clone repo
git clone https://github.com/your-username/deepfake-detection.git
cd deepfake-detection
2️⃣ Install dependencies
pip install -r requirements.txt
3️⃣ Start app
streamlit run app/app.py
📸 Sample Output
Prediction: FAKE 🔴
Confidence: 0.93
💡 Key Highlights
⚡ Real-time inference using Streamlit
👤 Robust face detection using MTCNN
🧠 Transfer learning for strong baseline performance
📦 Modular, production-style ML structure
🚀 Lightweight and deployable on CPU
🚧 Limitations
Works only on images (no video yet)
Performance depends on dataset quality
Needs GPU for large-scale training
🔮 Future Vision
🎥 Video deepfake detection (frame-by-frame analysis)
🔥 Grad-CAM explainability (see why model predicts fake)
🌐 REST API (FastAPI backend)
☁️ Cloud deployment (Streamlit Cloud / HuggingFace Spaces)
🧠 Upgrade to EfficientNet / Xception backbone
👨‍💻 Built By

A deep learning project focused on:

Computer Vision | AI Safety | Real-world ML Systems

📜 License

For educational and research use only.
