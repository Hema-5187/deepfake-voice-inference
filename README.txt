# 🎙️ Deepfake Voice Inference API

A production-ready FastAPI microservice for real-time deepfake voice detection using **Wav2Vec2 embeddings** and a **Support Vector Machine (SVM)** classifier.

This service accepts an audio file, extracts deep speech embeddings using Facebook's Wav2Vec2 model, and predicts whether the voice is **REAL** or **FAKE** along with confidence scores, performance metrics, and audio statistics.

---

## 🚀 Features

- 🎤 Upload WAV audio files
- 🧠 Wav2Vec2-based speech embedding extraction
- 🤖 SVM classifier for deepfake detection
- ⚡ FastAPI REST API
- 📊 Confidence score and prediction probability
- 📈 Audio statistics and inference performance metrics
- 📄 Interactive Swagger API documentation
- 🐳 Docker support for deployment
- ☁️ Ready for cloud deployment (Koyeb, Render, Railway, etc.)

---

## 🛠️ Tech Stack

- Python 3.11
- FastAPI
- Uvicorn
- PyTorch
- Hugging Face Transformers
- Scikit-learn
- Librosa
- Joblib
- Docker

---

## 📂 Project Structure

```
deepfake-voice-inference/
│
├── app/
│   ├── core/
│   ├── ml/
│   └── main.py
│
├── training/
│   └── models/
│       └── embedding_model.pkl
│
├── uploads/
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/<YOUR_USERNAME>/deepfake-voice-inference.git
cd deepfake-voice-inference
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the API

```bash
uvicorn app.main:app --reload
```

API will be available at

```
http://127.0.0.1:8000
```

Swagger Documentation

```
http://127.0.0.1:8000/docs
```

---

## 📡 API Endpoints

### GET /

Returns API information.

### GET /health

Returns service health status.

### POST /predict

Upload a WAV audio file for deepfake detection.

Example Response

```json
{
  "prediction": "REAL",
  "probability": 99.85,
  "confidence": "Very High",
  "message": "Prediction completed successfully.",
  "model": "Wav2Vec2 + SVM",
  "model_version": "2.0",
  "accuracy": 98.88,
  "feature_type": "Wav2Vec2 Embedding",
  "embedding_size": 768,
  "processing_time_ms": 6696.98,
  "performance": {
    "preprocessing_ms": 5549.54,
    "embedding_ms": 1133.71,
    "classification_ms": 13.45,
    "total_ms": 6696.98
  },
  "audio_statistics": {
    "duration_seconds": 3.94,
    "sample_rate": 16000,
    "num_samples": 62976,
    "rms_energy": 0.074769,
    "peak_amplitude": 1
  }
}
```

---

## 🐳 Docker

Build the Docker image

```bash
docker build -t deepfake-voice-inference .
```

Run the container

```bash
docker run -p 8000:8000 deepfake-voice-inference
```

---

## ☁️ Deployment

This project is ready to deploy using Docker on cloud platforms such as:

- Koyeb
- Render
- Railway
- Azure Container Apps
- Google Cloud Run
- AWS ECS

---

## 📈 Model Information

| Property | Value |
|----------|-------|
| Model | Wav2Vec2 + SVM |
| Feature Extraction | Facebook Wav2Vec2 |
| Embedding Size | 768 |
| Classifier | Support Vector Machine (SVM) |
| Accuracy | 98.88% |

---

## 📄 License

This project is released under the MIT License.

---

## 👩‍💻 Author

**Hema Maurya**

AI & Machine Learning Developer

GitHub: https://github.com/Hema-5187

# deepfake-voice-inference