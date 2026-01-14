# brain-tumor-detection
# 🧠 Brain Tumor Detection System

An AI-powered full-stack web application that uses deep learning to classify brain MRI scans into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor. Built with TensorFlow, FastAPI, and React.

![Project Banner](screenshots/banner.png)

## ⚠️ Important Disclaimer

**This project is for educational and research purposes only. It is NOT intended for clinical use and should not be used for actual medical diagnosis. Always consult qualified healthcare professionals for medical decisions.**

---

## 📋 Table of Contents

- [Features](#features)
- [Demo](#demo)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## ✨ Features

- **🎯 High Accuracy Classification**: 90%+ accuracy using EfficientNetB3 transfer learning
- **⚡ Real-time Predictions**: < 2 second response time for MRI scan analysis
- **🎨 Modern UI**: Responsive React interface with drag-and-drop file upload
- **📊 Detailed Results**: Confidence scores and probability distribution for all classes
- **🔒 Secure API**: FastAPI backend with input validation and error handling
- **🐳 Docker Support**: Containerized deployment for easy setup
- **📱 Responsive Design**: Works seamlessly on desktop and tablet devices
- **🔍 Explainable AI**: Visual confidence indicators and probability charts

---

## 🎬 Demo

### Live Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Video Demo
[Link to demo video if available]

---

## 🛠️ Tech Stack

### Machine Learning
- **TensorFlow 2.20.0** - Deep learning framework
- **Keras** - High-level neural networks API
- **EfficientNetB3** - Pre-trained CNN architecture
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Scikit-learn** - Evaluation metrics

### Backend
- **FastAPI** - Modern Python web framework
- **Uvicorn** - ASGI server
- **Python 3.9+** - Programming language
- **Pillow** - Image processing
- **OpenCV** - Computer vision operations

### Frontend
- **React 18** - JavaScript UI library
- **CSS3** - Styling
- **HTML5** - Markup

### DevOps
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

---

## 🏗️ Architecture