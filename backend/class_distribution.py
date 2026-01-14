"""
Check dataset class distribution
"""
import os

DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")

classes = ['glioma', 'meningioma', 'notumor', 'pituitary']

print("=" * 60)
print("DATASET CLASS DISTRIBUTION")
print("=" * 60)
print()

print("TRAINING SET:")
print("-" * 60)
train_counts = {}
total_train = 0

for cls in classes:
    cls_path = os.path.join(TRAIN_DIR, cls)
    if os.path.exists(cls_path):
        count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        train_counts[cls] = count
        total_train += count
        percentage = (count / total_train * 100) if total_train > 0 else 0
        print(f"{cls:15} {count:5} images")

print(f"\nTotal: {total_train} images")
print()

print("PERCENTAGES:")
for cls in classes:
    if cls in train_counts:
        percentage = (train_counts[cls] / total_train * 100)
        print(f"{cls:15} {percentage:5.1f}%")

print()
print("=" * 60)
print("TESTING SET:")
print("-" * 60)
test_counts = {}
total_test = 0

for cls in classes:
    cls_path = os.path.join(TEST_DIR, cls)
    if os.path.exists(cls_path):
        count = len([f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
        test_counts[cls] = count
        total_test += count
        print(f"{cls:15} {count:5} images")

print(f"\nTotal: {total_test} images")
print()

# Check for imbalance
max_count = max(train_counts.values())
min_count = min(train_counts.values())
imbalance_ratio = max_count / min_count

print("=" * 60)
print("ANALYSIS:")
print("-" * 60)
print(f"Imbalance Ratio: {imbalance_ratio:.2f}")

if imbalance_ratio > 1.5:
    print("⚠️  DATASET IS IMBALANCED!")
    print("   This can cause the model to prefer the majority class.")
    print("   Solution: Use class weights during training.")
else:
    print("✓ Dataset is reasonably balanced")

print()
print("=" * 60)