"""
Verify dataset is correctly loaded
"""
import os
from PIL import Image
import numpy as np

DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("=" * 60)
print("VERIFYING DATA LOADING")
print("=" * 60)
print()

for cls in classes:
    cls_path = os.path.join(TRAIN_DIR, cls)
    
    if not os.path.exists(cls_path):
        print(f"❌ {cls} folder NOT FOUND!")
        continue
    
    # Get first image
    images = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(images) == 0:
        print(f"❌ {cls} has NO images!")
        continue
    
    # Try loading first image
    img_path = os.path.join(cls_path, images[0])
    try:
        img = Image.open(img_path)
        img_array = np.array(img)
        
        print(f"✓ {cls:12} - {len(images)} images")
        print(f"  Sample: {images[0]}")
        print(f"  Shape: {img_array.shape}")
        print(f"  Type: {img.mode}")
        print()
    except Exception as e:
        print(f"❌ {cls} - Error loading image: {e}")
        print()

print("=" * 60)