"""
Simple Brain Tumor Classification Model Training
This script trains a CNN model to classify brain tumors
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("=" * 70)
print("BRAIN TUMOR CLASSIFICATION - MODEL TRAINING")
print("=" * 70)
print()

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16  # Small batch size for slower computers
EPOCHS = 20      # Start with fewer epochs for testing
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paths
DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_SAVE_PATH = "../models"

# Create models directory
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("Configuration:")
print(f"  • Image Size: {IMAGE_SIZE}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Epochs: {EPOCHS}")
print(f"  • Classes: {CLASSES}")
print()

# ========================
# STEP 1: DATA PREPARATION
# ========================
print("STEP 1: Preparing Data...")
print("-" * 70)

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of training data for validation
)

# Only rescaling for test data
test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")
print(f"✓ Test samples: {test_generator.samples}")
print()

# ========================
# STEP 2: BUILD MODEL
# ========================
print("STEP 2: Building Model...")
print("-" * 70)

# Load pre-trained EfficientNetB0
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# Freeze base model
base_model.trainable = False

# Build model
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ Model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")
print()

# ========================
# STEP 3: TRAIN MODEL
# ========================
print("STEP 3: Training Model...")
print("-" * 70)
print("This will take 15-30 minutes depending on your computer...")
print("You'll see progress bars for each epoch.")
print()

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'brain_tumor_model_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
]

# Train
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print()
print("✓ Training completed!")
print()

# ========================
# STEP 4: EVALUATE MODEL
# ========================
print("STEP 4: Evaluating Model...")
print("-" * 70)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)

print()
print(f"Test Results:")
print(f"  • Loss: {test_loss:.4f}")
print(f"  • Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print()

# ========================
# STEP 5: SAVE MODEL
# ========================
print("STEP 5: Saving Model...")
print("-" * 70)

final_model_path = os.path.join(MODEL_SAVE_PATH, 'brain_tumor_model_final.h5')
model.save(final_model_path)
print(f"✓ Model saved to: {final_model_path}")
print()

# ========================
# STEP 6: PLOT TRAINING HISTORY
# ========================
print("STEP 6: Creating Training Visualizations...")
print("-" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot accuracy
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot loss
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
print("✓ Training history saved as 'training_history.png'")
print()

# ========================
# SUMMARY
# ========================
print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print()
print("Summary:")
print(f"  • Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Model saved at: {final_model_path}")
print(f"  • Training history plot: training_history.png")
print()
print("Next steps:")
print("  1. Check the training_history.png to see learning curves")
print("  2. If accuracy is low (<90%), we can train for more epochs")
print("  3. Once satisfied, we'll build the web interface!")
print()
print("=" * 70)