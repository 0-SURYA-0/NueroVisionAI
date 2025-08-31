# 🧠 NueroVision AI

**Full-stack AI system for brain tumor detection from MRI scans with uncertainty quantification and explainable predictions.**

## 🔧 Tech Stack & Requirements

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-red)
![TorchVision](https://img.shields.io/badge/TorchVision-0.17+-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Flask](https://img.shields.io/badge/Flask-3.0+-lightgrey)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-brightgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-yellow)
![Scikit-learn](https://img.shields.io/badge/ScikitLearn-1.3+-orange)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- 🩻 **Multi-modal MRI fusion** (T1, T2, FLAIR) with normalization & augmentation  
- 🧩 **ResNet34 + Multi-Scale Self-Attention** for robust tumor feature extraction  
- 🎲 **Bayesian Convolutions + Monte Carlo Dropout** → Epistemic uncertainty  
- ⚖️ **Aleatoric head** → Data uncertainty  
- 🔥 **Focal Loss + KL Divergence regularization** with AdamW optimizer & Cosine Annealing  
- 🖼️ **Grad-CAM++ visualizations** for explainable predictions  
- ⚡ **FastAPI backend + React (Tailwind) frontend** for deployment  

---

## 🏗️ Folder Structure

```
NUEROVISION AI/
│── backend/                 # FastAPI backend (inference + APIs)
│   ├── main.py              # Entry point for FastAPI app
│   ├── routers/             # API routes
│   ├── models/              # Pydantic schemas, DB models
│   ├── services/            # Business logic, model serving
│   ├── utils/               # Helper functions
│   └── requirements.txt     # Backend dependencies
│
│── core/                    # Core ML pipeline
│   ├── __init__.py
│   ├── inference.py         # Model inference script
│   ├── model.py             # ResNet34 + Bayesian + Attention model
│   ├── train.py             # Training loop with Focal Loss + KL
│   ├── utils.py             # Preprocessing, metrics, helpers
│   └── requirements.txt     # Core ML dependencies
│
│── data/                    # MRI dataset (not pushed to repo)
│   ├── raw/                 # Raw MRI scans
│   ├── processed/           # Normalized & augmented data
│   └── splits/              # Train/Val/Test splits
│
│── frontend/                # React + Tailwind + Vite frontend
│   ├── src/
│   │   ├── components/      # UI components (UploadForm, Charts)
│   │   ├── pages/           # Dashboard, Results, About
│   │   ├── services/        # API calls to backend
│   │   └── assets/          # Logos, static files
│   ├── public/              # Public static files
│   └── package.json         # Frontend dependencies
│
│── notebooks/               # Jupyter notebooks (pipeline steps)
│   ├── 01_preprocessing.ipynb     # Data preprocessing
│   ├── 02_model_building.ipynb    # Model design & setup
│   ├── 03_training.ipynb          # Training experiments
│   ├── 04_evaluation.ipynb        # Model evaluation
│   ├── 05_visualizations.ipynb    # Results & Grad-CAM++ visualizations
│   └── 06_inference_demo.ipynb    # Demo inference with test scans
│
│── results/                 # Model outputs
│   ├── predictions/         # Prediction results
│   ├── heatmaps/            # Grad-CAM++ visualizations
│   └── metrics.json         # Accuracy, Loss, Uncertainty
│
│── .gitignore
│── LICENSE
│── requirements.txt         # Root-level dependencies
```

---

## 🛠️ Tech Stack

- **Machine Learning**: PyTorch, ResNet34, Bayesian CNN, Self-Attention  
- **Uncertainty Estimation**: Monte Carlo Dropout, Aleatoric Head  
- **Explainability**: Grad-CAM++  
- **Backend**: FastAPI  
- **Frontend**: React + Tailwind CSS + Vite  
- **Deployment**: REST API + Web UI  

---

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/0-SURYA-0/NueroVision-AI.git
cd NueroVision-AI
```

### 2️⃣ Backend Setup (FastAPI)

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

The inference API will be available at **http://localhost:8000**

### 3️⃣ Frontend Setup (React + Tailwind)

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at **http://localhost:5173**

### 4️⃣ Model Training (Optional)

```bash
cd core
pip install -r requirements.txt
python train.py
```

---

## 🚀 Usage

1. **Open the Frontend UI** at http://localhost:5173
2. **Upload an MRI scan** (T1/T2/FLAIR format)
3. **Get Results**:
   - ✅ Tumor probability score
   - 📉 Epistemic + Aleatoric uncertainty estimates
   - 🔥 Grad-CAM++ heatmap visualization

### API Endpoints

- `POST /predict` - Upload MRI scan for tumor detection
- `GET /health` - Health check endpoint
- `GET /docs` - Interactive API documentation

---

## 📊 Results

- **Accuracy**: (To be updated after training completion)
- **Uncertainty Calibration**: Reliable epistemic & aleatoric uncertainty estimates
- **Explainability**: Grad-CAM++ highlights tumor regions for medical interpretability
- **Performance**: Real-time inference with uncertainty quantification

---

## 🔬 Model Architecture

The system combines several advanced techniques:

- **Backbone**: ResNet34 with pre-trained weights
- **Attention Mechanism**: Multi-scale self-attention for feature refinement
- **Uncertainty Quantification**: 
  - Bayesian convolutions for epistemic uncertainty
  - Aleatoric head for data uncertainty
  - Monte Carlo dropout during inference
- **Loss Function**: Focal Loss + KL Divergence regularization
- **Optimization**: AdamW with Cosine Annealing scheduler

---

## 📁 Key Files

- `core/model.py` - Complete model architecture
- `core/train.py` - Training pipeline with uncertainty estimation
- `core/inference.py` - Inference script with Grad-CAM++
- `backend/main.py` - FastAPI application entry point
- `frontend/src/` - React frontend components

---

## 📈 Development Roadmap

- [ ] Deploy on cloud platform (Docker + AWS/GCP)
- [ ] Add support for additional MRI modalities (Diffusion, Perfusion)
- [ ] Implement clinical-grade user interface
- [ ] Add model performance benchmarking
- [ ] Integration with medical imaging standards (DICOM)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [BRATS Dataset](https://www.med.upenn.edu/sbia/brats2018.html) for brain tumor MRI scans
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [Grad-CAM++](https://arxiv.org/abs/1710.11063) for explainable AI visualizations
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [React](https://reactjs.org/) and [Tailwind CSS](https://tailwindcss.com/) for the frontend

---

## 📞 Contact

**Surya** - [GitHub Profile](https://github.com/0-SURYA-0)

Project Link: [https://github.com/0-SURYA-0/NueroVision-AI](https://github.com/0-SURYA-0/NueroVision-AI)

---

<div align="center">
  <strong>⭐ Star this repo if you found it helpful!</strong>
</div>
