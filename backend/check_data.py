"""
Check if dataset is downloaded and structured correctly
"""
import os

print("=" * 50)
print("CHECKING DATASET")
print("=" * 50)
print()

data_dir = "../data"
train_dir = os.path.join(data_dir, "Training")
test_dir = os.path.join(data_dir, "Testing")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

if not os.path.exists(data_dir):
    print("❌ Data directory not found!")
    print(f"   Expected at: {os.path.abspath(data_dir)}")
    exit()

print(f"✓ Data directory found: {os.path.abspath(data_dir)}")
print()

# Check Training data
if os.path.exists(train_dir):
    print("✓ Training directory found")
    for cls in classes:
        cls_path = os.path.join(train_dir, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  → {cls}: {count} images")
        else:
            print(f"  ❌ {cls} folder not found")
else:
    print("❌ Training directory not found!")

print()

# Check Testing data
if os.path.exists(test_dir):
    print("✓ Testing directory found")
    for cls in classes:
        cls_path = os.path.join(test_dir, cls)
        if os.path.exists(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  → {cls}: {count} images")
        else:
            print(f"  ❌ {cls} folder not found")
else:
    print("❌ Testing directory not found!")

print()
print("=" * 50)
print("✅ DATA CHECK COMPLETE!")
print("=" * 50)