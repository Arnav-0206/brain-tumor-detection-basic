# 🧠 NeuroScan AI - Brain Tumor Detection System

**The future of AI-powered medical imaging analysis**

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![License](https://img.shields.io/badge/license-MIT-blue)

---

## 🎯 Overview

NeuroScan AI is a cutting-edge brain tumor detection system that combines state-of-the-art deep learning with beautiful user experience and explainable AI. The system uses advanced models like EfficientNet-B4 and ResNet50 with transfer learning to provide accurate predictions while maintaining transparency through Grad-CAM visualizations.

### ✨ Key Features

- 🤖 **Modern AI Models**: EfficientNet-B4 / ResNet50 with transfer learning
- 🎨 **Beautiful UI**: React + TypeScript + Tailwind CSS with animations
- 🔍 **Explainable AI**: Grad-CAM visualizations (coming soon)
- 📝 **AI Narratives**: LLM-generated explanations (optional)
- ⚡ **Fast & Responsive**: Real-time predictions with smooth UX
- 🌙 **Dark Mode**: Modern glassmorphism design

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have:
- **Python 3.8+** installed ([Download here](https://www.python.org/downloads/))
- **Node.js 16+** and npm installed ([Download here](https://nodejs.org/))
- **Git** installed ([Download here](https://git-scm.com/))
- **Windows OS** (for batch scripts)

### One-Command Setup (Windows)

```bash
# Run the setup script
setup.bat
```

That's it! The script will:
- Create Python virtual environment
- Install all dependencies (backend + frontend)
- Setup configuration files
- Create data directories

### Running the Application

```bash
# Start both backend and frontend
run.bat
```

**Access:**
- 🎨 Frontend: http://localhost:3000
- ⚙️ Backend API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

---

## 📁 Project Structure

```
PROJECT/
├── backend/                      # FastAPI + PyTorch backend
│   ├── app/                     # API application
│   │   ├── main.py             # FastAPI app entry point
│   │   ├── config.py           # Configuration
│   │   └── routers/            # API endpoints
│   ├── ml/                      # ML models & training
│   │   ├── models/             # Model architectures
│   │   ├── training/           # Training scripts
│   │   ├── inference/          # Inference pipeline
│   │   ├── data/               # Data loaders
│   │   ├── explainability/     # Grad-CAM implementation
│   │   └── checkpoints/        # Saved model weights
│   ├── scripts/                 # Utility scripts
│   └── requirements.txt         # Python dependencies
├── frontend/                     # React + TypeScript frontend
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── utils/               # Utility functions
│   │   └── App.tsx              # Main app
│   ├── package.json             # Node dependencies
│   └── vite.config.ts           # Vite configuration
├── data/                         # Datasets
│   ├── raw/                     # Original dataset
│   │   ├── Training/           # Training images
│   │   └── Testing/            # Test images
│   └── processed/               # Processed data
├── Documentation/                # Project documentation
├── setup.bat                     # Setup script
├── run.bat                       # Run script
└── README.md                     # This file
```

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **ML**: PyTorch 2.x
- **Models**: EfficientNet-B4, ResNet50
- **Data**: Albumententations for augmentation
- **Training**: Early stopping, LR scheduling

### Frontend
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Animations**: Framer Motion
- **Upload**: React Dropzone
- **Icons**: Lucide React

### ML Pipeline
- Transfer learning with pretrained models
- Data augmentation (Albumentations)
- Stratified train/val/test splits
- Class-weighted loss for imbalanced data
- Cosine annealing LR scheduler

---

## 📊 Current Status

### ✅ Completed
- [x] Project structure & setup
- [x] Backend API foundation (FastAPI)
- [x] ML pipeline (data loading, models, training)
- [x] Frontend UI (React + TypeScript + Tailwind)
- [x] Upload interface with drag & drop
- [x] Results visualization
- [x] Animations & dark mode
- [x] Helper scripts (setup.bat, run.bat)

### 🔄 In Progress
- [ ] Dataset download & preparation
- [ ] Model training
- [ ] Inference API endpoints
- [ ] Grad-CAM implementation

### 📋 Planned
- [ ] LLM integration for narratives
- [ ] Model comparison dashboard
- [ ] Batch processing
- [ ] Deployment (Docker + Cloud)

---

## 💻 Development

### Detailed Setup Instructions

#### Step 1: Clone the Repository
```bash
git clone https://github.com/Arnav-0206/brain-tumor-detection-basic.git
cd brain-tumor-detection-basic
```

#### Step 2: Setup Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

#### Step 3: Setup Frontend
```bash
cd frontend
npm install
```

#### Step 4: Prepare Dataset
Place your brain tumor dataset in the following structure:
```
data/raw/
├── Training/
│   ├── glioma_tumor/
│   ├── meningioma_tumor/
│   ├── no_tumor/
│   └── pituitary_tumor/
└── Testing/
    ├── glioma_tumor/
    ├── meningioma_tumor/
    ├── no_tumor/
    └── pituitary_tumor/
```

### Running Manually

**Terminal 1 - Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

---

## 📚 Documentation

For more detailed information, check out:

### Quick Start
- [Scripts Guide](SCRIPTS.md) - Setup and run scripts
- [Dataset Guide](data/DATASET_GUIDE.md) - Download and prepare dataset

### Backend
- [Backend Setup](backend/SETUP.md) - Detailed backend setup
- [API Guide](backend/API_GUIDE.md) - API endpoints documentation
- [Backend README](backend/README.md) - Backend overview

### Frontend  
- [Frontend README](frontend/README.md) - Frontend documentation

### Technical Details
- [Complete Overview](Documentation/COMPLETE_PROJECT_OVERVIEW.md) - Full project architecture
- [Technical Documentation](Documentation/DETAILED_TECHNICAL_DOCUMENTATION.md) - In-depth technical details
- [Quick Reference](Documentation/QUICK_REFERENCE.md) - Quick command reference

---

## 🎯 Project Highlights

What makes NeuroScan AI stand out:

1. **Modern Architecture**: Latest ML models with proven performance
2. **Beautiful UX**: Professional UI with smooth animations
3. **Explainable AI**: Grad-CAM visualizations for transparency
4. **AI Narratives**: LLM-generated explanations for better understanding
5. **Easy Setup**: One-command installation for quick deployment
6. **Professional Code**: Built with TypeScript, best practices, and proper documentation

---

## ⚠️ Important Notes

- This is a **research/educational project**
- **Not for clinical use** or medical diagnosis
- Always consult medical professionals for health concerns
- Dataset used is for demonstration purposes

---

## 🏆 Why Choose NeuroScan AI?

This project delivers:
- ✅ **Technical Excellence**: Modern ML architecture with proven performance
- ✅ **Superior UX**: Intuitive interface with smooth, responsive design
- ✅ **Transparency**: Explainable AI with visual interpretations
- ✅ **Production Ready**: Comprehensive documentation and professional codebase

---

## 📝 License

MIT License - feel free to use for learning and research!

---

## 🙏 Acknowledgments

- Brain tumor datasets from Kaggle community
- Open source ML frameworks (PyTorch, timm)
- React & modern web ecosystem

---

**Built with ❤️ and lots of ☕**

🚀 Ready to revolutionize medical AI!
