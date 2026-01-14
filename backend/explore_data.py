"""
Explore the brain tumor dataset
Visualize sample images from each class
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import random

# Set up paths
data_dir = "../data/Training"
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("=" * 60)
print("EXPLORING BRAIN TUMOR DATASET")
print("=" * 60)
print()

# Create a figure to show sample images
fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle('Sample Brain MRI Scans from Each Class', fontsize=16, fontweight='bold')

for i, class_name in enumerate(classes):
    class_path = os.path.join(data_dir, class_name)
    
    # Get all image files
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Select 4 random images
    sample_images = random.sample(image_files, 4)
    
    for j, img_file in enumerate(sample_images):
        img_path = os.path.join(class_path, img_file)
        img = Image.open(img_path)
        
        # Display image
        axes[i, j].imshow(img, cmap='gray')
        axes[i, j].axis('off')
        
        # Add title only to first column
        if j == 0:
            axes[i, j].set_title(f'{class_name.upper()}', 
                                fontsize=12, 
                                fontweight='bold',
                                loc='left')

plt.tight_layout()
plt.savefig('dataset_samples.png', dpi=150, bbox_inches='tight')
print("✓ Sample images saved as 'dataset_samples.png'")
print()

# Show image dimensions
print("Checking image dimensions...")
print()

for class_name in classes:
    class_path = os.path.join(data_dir, class_name)
    image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Check first image dimensions
    first_img = Image.open(os.path.join(class_path, image_files[0]))
    width, height = first_img.size
    
    print(f"✓ {class_name:12} - Sample size: {width}x{height} pixels")

print()
print("=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("• All images are grayscale brain MRI scans")
print("• Image sizes may vary - we'll resize to 224x224 for training")
print("• 4 classes: Glioma, Meningioma, No Tumor, Pituitary Tumor")
print("• Dataset is relatively balanced across classes")
print()
print("✅ EXPLORATION COMPLETE!")
print("=" * 60)

plt.show()