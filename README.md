# 🎙️ Speech Emotion Recognition using Deep Learning

> Deep learning system that classifies 8 human emotions from audio using CNN, achieving 92% accuracy on the RAVDESS dataset.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep_Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

---

## 📌 Overview

This project builds an **emotion classification system** from speech audio. It evaluates multiple ML models — SVM, MLP, and CNN — on the RAVDESS dataset containing 8 distinct emotions. The CNN model achieved the highest accuracy of **92%**, outperforming classical ML approaches significantly.

---

## 🎭 Emotions Classified

| Label | Emotion |
|-------|---------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## ✨ Features

- 🎵 **RAVDESS Dataset** — 24 professional actors, 8 emotions, 1440+ audio files
- 🧠 **Multi-Model Comparison** — Benchmarks SVM, MLP, and CNN architectures
- 📈 **92% CNN Accuracy** — Superior performance validated with precision, recall, F1
- 🔊 **MFCC Feature Extraction** — Mel-Frequency Cepstral Coefficients from audio
- 📊 **Comprehensive Metrics** — Confusion matrix, classification report, accuracy curves

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.8+ |
| Deep Learning | CNN (Keras/TensorFlow) |
| Classical ML | SVM, MLP (Scikit-learn) |
| Audio Processing | librosa |
| Feature Extraction | MFCC, Chroma, Mel Spectrogram |
| Visualization | Matplotlib, Seaborn |
| Data | NumPy, Pandas |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Gangadhar-tech-coder/speech-emotion-recognition.git
cd speech-emotion-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download RAVDESS dataset
# Place audio files in: data/ravdess/

# 4. Run feature extraction
python extract_features.py

# 5. Train models
python train.py

# 6. Evaluate all models
python evaluate.py
```

---

## 📁 Project Structure

```
speech-emotion-recognition/
├── data/
│   └── ravdess/             # RAVDESS audio dataset
├── features/
│   └── features.pkl         # Extracted MFCC features
├── models/
│   ├── cnn_model.h5         # Saved CNN model
│   ├── svm_model.pkl        # Saved SVM model
│   └── mlp_model.pkl        # Saved MLP model
├── extract_features.py      # MFCC feature extraction
├── train.py                 # Training script for all models
├── evaluate.py              # Model evaluation + metrics
├── predict.py               # Single-file emotion prediction
├── requirements.txt
└── README.md
```

---

## 📊 Model Performance Comparison

| Model | Accuracy | Notes |
|-------|----------|-------|
| SVM | ~72% | Good baseline, fast training |
| MLP | ~78% | Better than SVM, moderate complexity |
| **CNN** | **92%** | Best performance, captures temporal patterns |

---

## 🧪 How It Works

```
Audio File → Preprocessing → MFCC Extraction → Model Input
                                                    ↓
                                           CNN / SVM / MLP
                                                    ↓
                                         Emotion Prediction
```

1. Load `.wav` audio file from RAVDESS
2. Extract MFCC, Chroma, and Mel Spectrogram features using `librosa`
3. Feed features into trained model
4. Output predicted emotion with confidence score

---

## 🔮 Predict on Custom Audio

```python
from predict import predict_emotion

result = predict_emotion("your_audio.wav")
print(f"Predicted Emotion: {result}")
```

---
