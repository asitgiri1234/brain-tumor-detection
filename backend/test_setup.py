"""
Test script to verify installation
"""
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

print("=" * 50)
print("TESTING SETUP")
print("=" * 50)
print()

print("✓ All imports successful!")
print(f"✓ TensorFlow version: {tf.__version__}")
print(f"✓ NumPy version: {np.__version__}")
print(f"✓ Pandas version: {pd.__version__}")
print()

# Check if GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✓ GPU FOUND: {len(gpus)} GPU(s) available")
    print("  → Training will be FAST! 🚀")
else:
    print("✓ No GPU found - will use CPU")
    print("  → Training will be slower but works fine")

print()
print("=" * 50)
print("✅ SETUP SUCCESSFUL! READY FOR NEXT STEP.")
print("=" * 50)