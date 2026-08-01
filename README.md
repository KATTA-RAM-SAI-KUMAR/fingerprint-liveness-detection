# 🔒 FingerSecure AI – Fingerprint Liveness Detection using Deep Learning

## 📌 Overview

FingerSecure AI is a deep learning-based Fingerprint Presentation Attack Detection (PAD) system designed to distinguish between genuine (live) fingerprints and spoof fingerprints. The system helps improve biometric security by detecting presentation attacks such as printed fingerprints and screen-displayed fingerprint images.

The project uses Transfer Learning with MobileNetV3-Small for efficient and accurate real-time fingerprint liveness detection.

---

## ✨ Features

- Real-time Fingerprint Liveness Detection
- Transfer Learning using MobileNetV3-Small
- Web Interface built with Flask
- Balanced Dataset Preparation
- Automatic Spoof Dataset Generation
- Early Stopping
- Best Model Checkpoint Saving
- Learning Rate Scheduling
- APCER, BPCER and EER Evaluation
- Confusion Matrix and Training Curves
- Fast Inference (~5 ms per image)

---

## 📂 Project Structure

```
fingerprint-liveness-detection/
│
├── app.py
├── liveness_train.py
├── liveness_eval.py
├── liveness_infer.py
├── prepare_dataset.py
├── generate_spoof_dataset.py
│
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
│
├── data/
│   ├── live/
│   ├── spoof/
│   └── SOCOFing/
│
├── models/
│   ├── checkpoints/
│   └── liveness_model.pth
│
├── outputs/
│
├── static/
│
├── templates/
│
├── utils/
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── metrics.py
│   └── plots.py
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model

- MobileNetV3-Small
- Transfer Learning
- CrossEntropy Loss
- Adam Optimizer
- ReduceLROnPlateau Scheduler
- Early Stopping
- Best Validation Model Saving

---

## 📊 Dataset

### Live Images

- SOCOFing Real Fingerprints

### Spoof Images

Spoof samples include realistic presentation attacks such as:

- Printed fingerprints
- Screen display attacks
- Camera captured attacks

The dataset is automatically balanced before training.

---

## 🚀 Training

Run:

```bash
python liveness_train.py
```

The best model is automatically saved to:

```
models/checkpoints/best_model.pth
```

Final model:

```
models/liveness_model.pth
```

---

## 📈 Evaluation

Run

```bash
python liveness_eval.py
```

Evaluation includes:

- Accuracy
- Precision
- Recall
- F1 Score
- APCER
- BPCER
- EER
- Confusion Matrix
- Score Distribution
- Average Inference Time

---

## 🌐 Run Web Application

```bash
python app.py
```

Open

```
http://127.0.0.1:5000
```

Upload a fingerprint image and the system predicts:

- LIVE
- SPOOF

along with

- Confidence Score
- Inference Time

---

## 📊 Results

| Metric | Value |
|---------|-------|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |
| APCER | 0.0000 |
| BPCER | 0.0000 |
| EER | 0.0000 |
| Average Inference | ~5 ms |

---

## 🔮 Future Improvements

- Support additional spoof attack types
- Test on multiple fingerprint sensors
- Real-time webcam integration
- ONNX/TensorRT optimization
- Mobile deployment

---

## 👨‍💻 Technologies Used

- Python
- PyTorch
- TorchVision
- OpenCV
- Flask
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

---

## 📄 License

This project is developed for educational and evaluation purposes.