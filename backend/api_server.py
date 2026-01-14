"""
FastAPI Backend for Brain Tumor Detection
Serves predictions from the trained model
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import os

print("=" * 70)
print("BRAIN TUMOR DETECTION API")
print("=" * 70)
print()

# Initialize FastAPI
app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for classifying brain MRI scans",
    version="1.0.0"
)

# Enable CORS (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor']
IMAGE_SIZE = (224, 224)

# ========================
# LOAD MODEL ON STARTUP
# ========================
@app.on_event("startup")
async def load_model():
    """Load the trained model when server starts"""
    global model
    
    # Try to find the model
    possible_paths = [
        "../models/brain_tumor_model_improved.h5",
        "../models/model_phase2_best.h5",
        "../models/brain_tumor_model_final.h5"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ ERROR: No trained model found!")
        print("   Looked in:")
        for path in possible_paths:
            print(f"   - {path}")
        print("\n   Please train the model first!")
        return
    
    try:
        print(f"Loading model from: {model_path}")
        model = keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        print(f"   Model input shape: {model.input_shape}")
        print(f"   Model output shape: {model.output_shape}")
        print()
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        print()

# ========================
# HELPER FUNCTIONS
# ========================
def preprocess_image(image_bytes):
    """Preprocess image for model prediction"""
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize(IMAGE_SIZE)
        
        # Convert to array and normalize
        image_array = np.array(image)
        image_array = image_array.astype(np.float32) / 255.0
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
    
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing image: {str(e)}"
        )

# ========================
# API ENDPOINTS
# ========================
@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Brain Tumor Detection API",
        "status": "healthy" if model is not None else "model not loaded",
        "model_loaded": model is not None,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict brain tumor type from MRI scan
    
    Args:
        file: Image file (JPG, PNG)
    
    Returns:
        JSON with prediction results
    """
    
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and ensure model is trained."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPG, PNG, etc.)"
        )
    
    try:
        # Read file contents
        contents = await file.read()
        print(f"Received image: {file.filename} ({len(contents)} bytes)")
        
        # Preprocess image
        image_array = preprocess_image(contents)
        print(f"Image preprocessed: shape {image_array.shape}")
        
        # Make prediction
        predictions = model.predict(image_array, verbose=0)[0]
        print(f"Raw predictions: {predictions}")
        
        # Get predicted class
        predicted_class_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx])
        
        # All probabilities
        all_probabilities = {
            class_name: float(prob)
            for class_name, prob in zip(CLASS_NAMES, predictions)
        }
        
        print(f"Prediction: {predicted_class} ({confidence:.2%})")
        print()
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "all_probabilities": all_probabilities
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during prediction: {str(e)}"
        )

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "classes": CLASS_NAMES,
        "input_shape": [*IMAGE_SIZE, 3],
        "num_classes": len(CLASS_NAMES),
        "model_loaded": True
    }

# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    import uvicorn
    
    print("Starting server at http://localhost:8000")
    print("API documentation at http://localhost:8000/docs")
    print()
    print("Press CTRL+C to stop the server")
    print("=" * 70)
    print()
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )