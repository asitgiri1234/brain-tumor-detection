"""
Improved Brain Tumor Classification Training
Better settings for higher accuracy
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("BRAIN TUMOR CLASSIFICATION - IMPROVED TRAINING")
print("=" * 70)
print()

# IMPROVED Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32  # Increased for better learning
EPOCHS = 50      # More epochs
LEARNING_RATE = 0.001
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paths
DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_SAVE_PATH = "../models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("Improved Configuration:")
print(f"  • Image Size: {IMAGE_SIZE}")
print(f"  • Batch Size: {BATCH_SIZE}")
print(f"  • Epochs: {EPOCHS}")
print(f"  • Learning Rate: {LEARNING_RATE}")
print()

# ========================
# DATA PREPARATION with MORE AUGMENTATION
# ========================
print("STEP 1: Preparing Data with Augmentation...")
print("-" * 70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,      # Increased
    width_shift_range=0.2,  # Increased
    height_shift_range=0.2, # Increased
    shear_range=0.2,        # Added
    zoom_range=0.2,         # Increased
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255)

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
print(f"✓ Class indices: {train_generator.class_indices}")
print()

# ========================
# BUILD IMPROVED MODEL
# ========================
print("STEP 2: Building Improved Model...")
print("-" * 70)

base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

# INITIALLY freeze base model
base_model.trainable = False

# Build model with MORE layers
inputs = keras.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)  # Increased neurons
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("✓ Model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")
print()

# ========================
# PHASE 1: TRAIN TOP LAYERS
# ========================
print("STEP 3: PHASE 1 - Training Top Layers...")
print("-" * 70)
print("Training only the classification head (faster)")
print()

callbacks_phase1 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'model_phase1_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,  # First 25 epochs
    callbacks=callbacks_phase1,
    verbose=1
)

print()
print("✓ Phase 1 completed!")
print()

# ========================
# PHASE 2: FINE-TUNE ENTIRE MODEL
# ========================
print("STEP 4: PHASE 2 - Fine-Tuning Entire Model...")
print("-" * 70)
print("Unfreezing base model for fine-tuning")
print()

# Unfreeze the base model
base_model.trainable = True

# Freeze first 100 layers (keep early features stable)
for layer in base_model.layers[:100]:
    layer.trainable = False

print(f"✓ Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompile with LOWER learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),  # 10x lower
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

callbacks_phase2 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'model_phase2_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=25,  # Another 25 epochs
    callbacks=callbacks_phase2,
    verbose=1
)

print()
print("✓ Phase 2 completed!")
print()

# ========================
# EVALUATE
# ========================
print("STEP 5: Evaluating Final Model...")
print("-" * 70)

test_results = model.evaluate(test_generator, verbose=1)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_precision = test_results[2]
test_recall = test_results[3]

print()
print(f"Final Test Results:")
print(f"  • Loss: {test_loss:.4f}")
print(f"  • Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Precision: {test_precision:.4f}")
print(f"  • Recall: {test_recall:.4f}")
print()

# ========================
# SAVE FINAL MODEL
# ========================
final_model_path = os.path.join(MODEL_SAVE_PATH, 'brain_tumor_model_improved.h5')
model.save(final_model_path)
print(f"✓ Final model saved to: {final_model_path}")
print()

# ========================
# PLOT TRAINING HISTORY
# ========================
print("STEP 6: Creating Visualizations...")
print("-" * 70)

# Combine both phases
combined_accuracy = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
combined_val_accuracy = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
combined_loss = history_phase1.history['loss'] + history_phase2.history['loss']
combined_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(combined_accuracy, label='Training Accuracy', linewidth=2, color='#0066CC')
ax1.plot(combined_val_accuracy, label='Validation Accuracy', linewidth=2, color='#00A3E0')
ax1.axvline(x=25, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
ax1.set_title('Model Accuracy (Two-Phase Training)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

# Loss plot
ax2.plot(combined_loss, label='Training Loss', linewidth=2, color='#0066CC')
ax2.plot(combined_val_loss, label='Validation Loss', linewidth=2, color='#00A3E0')
ax2.axvline(x=25, color='red', linestyle='--', label='Fine-tuning starts', alpha=0.7)
ax2.set_title('Model Loss (Two-Phase Training)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_improved.png', dpi=150, bbox_inches='tight')
print("✓ Training history saved as 'training_history_improved.png'")
print()

# ========================
# SUMMARY
# ========================
print("=" * 70)
print("✅ IMPROVED TRAINING COMPLETE!")
print("=" * 70)
print()
print("Summary:")
print(f"  • Final Test Accuracy: {test_accuracy*100:.2f}%")
print(f"  • Final Test Precision: {test_precision*100:.2f}%")
print(f"  • Final Test Recall: {test_recall*100:.2f}%")
print(f"  • Model saved at: {final_model_path}")
print()

if test_accuracy >= 0.90:
    print("🎉 EXCELLENT! Accuracy >= 90%")
    print("   Ready to proceed to web interface!")
elif test_accuracy >= 0.80:
    print("✓ GOOD! Accuracy >= 80%")
    print("   Consider training a bit more for better results")
else:
    print("⚠ Accuracy still low. Possible issues:")
    print("   1. Check if data is correctly organized")
    print("   2. Try training even longer")
    print("   3. Verify images are loading properly")

print()
print("=" * 70)