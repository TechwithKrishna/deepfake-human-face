# DeepShield AI: Deepfake Detection System

> **Real vs Fake Face Detection using Deep Learning**

## Overview

DeepShield AI is a production-ready deep learning system for detecting AI-generated faces and deepfakes in digital images. As deepfake technology becomes increasingly sophisticated, reliable detection mechanisms are essential for maintaining visual authenticity and preventing misuse.

This project implements a lightweight, CPU-deployable solution combining advanced face detection and deep learning classification to provide real-time predictions with confidence scoring.

---

## Problem Statement

Deepfakes and AI-generated faces present significant challenges across multiple domains:

- **Identity Fraud**: Spoofing authentication systems and impersonating individuals
- **Misinformation**: Creating false evidence and spreading disinformation
- **Security Threats**: Compromising trust in digital communication and media
- **Legal/Compliance**: Regulatory requirements for content authenticity verification

DeepShield AI provides a robust baseline solution for addressing these challenges in security and content verification workflows.

---

## System Architecture

```
Image Input (JPG/PNG)
        ↓
MTCNN Face Detection & Localization
        ↓
Face Extraction & Preprocessing (224x224)
        ↓
ResNet18 Feature Extraction
        ↓
Binary Classification (Real/Fake)
        ↓
Confidence Score & Output
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Face Detection** | MTCNN | Robust facial region localization |
| **Feature Extraction** | ResNet18 | Transfer learning backbone |
| **Classification** | Sigmoid Output | Binary classification with probability |
| **Interface** | Streamlit | Real-time inference web application |
| **Acceleration** | PyTorch | Optimized deep learning framework |

---

## Technical Specifications

### Model Configuration

| Parameter | Value |
|-----------|-------|
| **Backbone Architecture** | ResNet18 |
| **Training Approach** | Transfer Learning |
| **Input Resolution** | 224 × 224 pixels |
| **Output Layer** | Sigmoid activation |
| **Classification Labels** | 0 (Fake) / 1 (Real) |
| **Inference Framework** | PyTorch |

### Performance Characteristics

- **Inference Speed**: Real-time processing on CPU
- **Memory Footprint**: Lightweight model (~50MB)
- **Deployment**: CPU-compatible, minimal hardware requirements
- **Output**: Binary classification + confidence score (0-1)

---

## Tech Stack

### Core Dependencies

```
Python 3.8+
PyTorch 1.9+           # Deep Learning Framework
MTCNN                  # Face Detection
NumPy                  # Numerical Computing
Pillow                 # Image Processing
Streamlit              # Web Application
scikit-learn           # Utilities
```

### Dataset

**Source**: [Deepfake Face Images Dataset (Kaggle)](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images)

**Characteristics**:
- Balanced binary classification dataset
- Real human face images
- AI-generated and manipulated face images
- Pre-organized train/test splits

---

## Project Structure

```
deepfake-human-face/
│
├── app/
│   └── app.py                  # Streamlit web interface
│
├── src/
│   ├── model.py                # Model architecture definition
│   ├── predict.py              # Inference pipeline
│   ├── preprocessing.py        # Image preprocessing utilities
│   └── utils.py                # Helper functions
│
├── models/
│   └── deepshield_resnet18.pth # Trained model weights
│
├── notebooks/
│   └── training.ipynb          # Experimentation & training logs
│
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation
└── .gitignore
```

---

## Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 500MB disk space (model + dependencies)

### Step 1: Clone Repository

```bash
git clone https://github.com/TechwithKrishna/deepfake-human-face.git
cd deepfake-human-face
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
streamlit run app/app.py
```

The application will launch at `http://localhost:8501`

---

## Usage

### Web Interface (Recommended)

1. Launch the Streamlit application
2. Upload an image (JPG, PNG, or BMP format)
3. Click "Analyze" to process
4. View results:
   - **Classification**: REAL or FAKE
   - **Confidence Score**: 0.0 - 1.0 probability
   - **Processing Time**: Inference duration

### Programmatic Usage

```python
from src.predict import DeepfakeDetector

detector = DeepfakeDetector(model_path='models/deepshield_resnet18.pth')
result = detector.predict('sample_image.jpg')

print(f"Classification: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## Model Performance

### Training Details

- **Training Type**: Transfer Learning on ResNet18
- **Data Split**: 80% train / 20% test
- **Optimization**: Adam optimizer with learning rate scheduling
- **Loss Function**: Binary Cross-Entropy
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score

*Detailed performance metrics available in `notebooks/training.ipynb`*

---

## Limitations & Considerations

### Current Limitations

1. **Image-Only Processing**: Video analysis not yet implemented
2. **Dataset Dependency**: Performance varies with image quality and dataset diversity
3. **Computational Constraints**: Large-scale batch processing requires GPU acceleration
4. **Face Detection Prerequisite**: Requires detectable frontal face (MTCNN dependency)

### Model Behavior

- Best performance on clear, frontal facial images
- May struggle with extreme angles, occlusion, or low resolution
- Confidence scores reflect model certainty, not ground truth reliability
- Dataset bias may affect performance on underrepresented demographics

---

## Future Roadmap

### Near-term Enhancements

- [ ] Video deepfake detection with frame-by-frame analysis
- [ ] Grad-CAM explainability visualization
- [ ] Multi-model ensemble approach for robustness
- [ ] Performance metrics dashboard

### Medium-term Goals

- [ ] REST API development (FastAPI)
- [ ] Cloud deployment (Streamlit Cloud, HuggingFace Spaces)
- [ ] Model architecture upgrades (EfficientNet, Xception)
- [ ] Batch processing capabilities

### Long-term Vision

- [ ] Real-time video streaming support
- [ ] Adversarial robustness evaluation
- [ ] Deployment on edge devices
- [ ] Integration with security systems

---

## Contributing

Contributions are welcome. Please follow standard GitHub workflow:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open Pull Request

---

## Citation

If you use this project in research or production, please cite:

```bibtex
@software{deepshield_2024,
  author = {TechwithKrishna},
  title = {DeepShield AI: Deepfake Detection System},
  year = {2024},
  url = {https://github.com/TechwithKrishna/deepfake-human-face}
}
```

---

## License

This project is provided for **educational and research purposes only**.

**Disclaimer**: This tool is intended to support content verification workflows. Users are responsible for legal compliance when applying this technology. The authors assume no liability for misuse.

---

## Contact & Support

- **Repository**: [GitHub - TechwithKrishna/deepfake-human-face](https://github.com/TechwithKrishna/deepfake-human-face)
- **Issues**: Report bugs via GitHub Issues
- **Feedback**: Suggestions and improvements welcome

---

## References

- MTCNN: [Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks](https://arxiv.org/abs/1604.02878)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Transfer Learning in Computer Vision: [Fine-tuning Deep Convolutional Networks](https://cs231n.github.io/transfer-learning/)

---

**Last Updated**: 2026-05-03
