Brain Tumor Detection

A deep learning–based system that classifies brain MRI images into four categories: Glioma, Meningioma, Pituitary Tumor, and No Tumor.

The project uses a convolutional neural network trained with transfer learning and provides predictions through a simple backend API with a lightweight frontend interface.

⚠️ Disclaimer: This project is for educational and research purposes only and must not be used for real-world medical diagnosis.

🚀 Features

Brain MRI image classification

Transfer learning–based CNN model

FastAPI backend for predictions

Simple frontend for uploading MRI images

Confidence score for each prediction

🛠 Tech Stack

Python

TensorFlow / Keras

FastAPI

React

NumPy, OpenCV

📁 Project Structure

brain-tumor-detection/
├── backend/ – API and model logic
├── frontend/ – React user interface
├── models/ – Trained model files
├── data/ – Dataset (not included)
├── README.md
└── .gitignore

▶️ How to Run
Backend

Navigate to the backend folder

Install dependencies from requirements.txt

Start the FastAPI server

Frontend

Navigate to the frontend folder

Install dependencies using npm

Start the React development server

📊 Model Overview

Architecture: EfficientNet (transfer learning)

Input: Brain MRI images

Output: Tumor class with confidence score

Test accuracy: ~90%

🔮 Future Improvements

Grad-CAM visualization

DICOM image support

Model optimization

Cloud deployment