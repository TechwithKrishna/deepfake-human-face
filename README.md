<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DeepShield AI</title>

<style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        background: #0f172a;
        color: #e2e8f0;
        line-height: 1.6;
    }

    .container {
        max-width: 900px;
        margin: auto;
        padding: 40px;
    }

    h1 {
        font-size: 40px;
        color: #38bdf8;
    }

    h2 {
        color: #60a5fa;
        margin-top: 30px;
    }

    .tagline {
        font-size: 18px;
        color: #94a3b8;
    }

    .card {
        background: #1e293b;
        padding: 20px;
        border-radius: 12px;
        margin-top: 20px;
    }

    .highlight {
        color: #22c55e;
        font-weight: bold;
    }

    .danger {
        color: #ef4444;
        font-weight: bold;
    }

    a {
        color: #38bdf8;
        text-decoration: none;
    }

    code {
        background: #0f172a;
        padding: 3px 6px;
        border-radius: 5px;
    }

    .footer {
        margin-top: 40px;
        font-size: 14px;
        color: #64748b;
    }
</style>

</head>

<body>

<div class="container">

    <h1>🕵️‍♂️ DeepShield AI</h1>
    <p class="tagline">
        Real vs Fake Face Detection using Deep Learning
    </p>

    <div class="card">
        <p>
            <b>“If you can’t trust what you see, what can you trust?”</b><br><br>
            DeepShield AI detects whether a human face is <b>REAL or FAKE</b> using deep learning.
        </p>
    </div>

    <h2>🚀 System Output</h2>
    <div class="card">
        🟢 <span class="highlight">REAL</span> — Authentic Human Face<br>
        🔴 <span class="danger">FAKE</span> — AI Generated Face<br>
        📊 Confidence Score included
    </div>

    <h2>⚡ Why this matters</h2>
    <div class="card">
        Deepfakes are now used in:
        <ul>
            <li>🎭 Identity fraud</li>
            <li>📰 Misinformation</li>
            <li>🔐 Security threats</li>
        </ul>
        This system helps restore visual trust.
    </div>

    <h2>🧠 Pipeline</h2>
    <div class="card">
        Image Upload → MTCNN Face Detection → Cropping → ResNet Model → Prediction
    </div>

    <h2>🏗️ Tech Stack</h2>
    <div class="card">
        Python • PyTorch • ResNet18 • MTCNN • Streamlit
    </div>

    <h2>📊 Dataset</h2>
    <div class="card">
        Deepfake Face Images Dataset from Kaggle:<br><br>
        🔗 <a href="https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images" target="_blank">
        https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images
        </a>
    </div>

    <h2>🖥️ Run Project</h2>
    <div class="card">
        <code>git clone repo-url</code><br>
        <code>pip install -r requirements.txt</code><br>
        <code>streamlit run app/app.py</code>
    </div>

    <h2>🔮 Future Work</h2>
    <div class="card">
        🎥 Video deepfake detection<br>
        🔥 Grad-CAM explainability<br>
        ☁️ Cloud deployment<br>
        🧠 Better models (EfficientNet / Xception)
    </div>

    <div class="footer">
        Built for AI Safety • Computer Vision • Deep Learning
    </div>

</div>

</body>
</html>
