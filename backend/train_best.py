"""
Best Training Strategy for High Accuracy
Goal: 90%+ accuracy with balanced predictions
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3  # Bigger model!
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

print("=" * 70)
print("HIGH ACCURACY TRAINING - BEST SETTINGS")
print("=" * 70)
print()

# BEST Configuration
IMAGE_SIZE = (300, 300)  # Larger images = more detail
BATCH_SIZE = 16          # Smaller batch = better learning
EPOCHS = 60              # More epochs
LEARNING_RATE = 0.0005   # Lower learning rate
CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Paths
DATA_DIR = "../data"
TRAIN_DIR = os.path.join(DATA_DIR, "Training")
TEST_DIR = os.path.join(DATA_DIR, "Testing")
MODEL_SAVE_PATH = "../models"

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

print("BEST Configuration:")
print(f"  • Image Size: {IMAGE_SIZE} (larger for more detail)")
print(f"  • Batch Size: {BATCH_SIZE} (smaller for better learning)")
print(f"  • Epochs: {EPOCHS} (more training)")
print(f"  • Model: EfficientNetB3 (more powerful)")
print()

# ========================
# DATA PREPARATION
# ========================
print("STEP 1: Preparing Data with Strong Augmentation...")
print("-" * 70)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
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
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
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
# BUILD BETTER MODEL
# ========================
print("STEP 2: Building Larger Model (EfficientNetB3)...")
print("-" * 70)

base_model = EfficientNetB3(  # Bigger than B0!
    weights='imagenet',
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

base_model.trainable = False

inputs = keras.Input(shape=(*IMAGE_SIZE, 3))

# Data augmentation in model
x = inputs

# Base model
x = base_model(x, training=False)

# Custom head
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.0001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

print("✓ Model built successfully!")
print(f"✓ Total parameters: {model.count_params():,}")
print()

# ========================
# PHASE 1: WARM-UP
# ========================
print("STEP 3: PHASE 1 - Warm-up Training (30 epochs)...")
print("-" * 70)

callbacks_phase1 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'best_model_phase1.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

history_phase1 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=callbacks_phase1,
    verbose=1
)

print()
print("✓ Phase 1 completed!")
val_acc_phase1 = max(history_phase1.history['val_accuracy'])
print(f"✓ Best validation accuracy: {val_acc_phase1:.4f} ({val_acc_phase1*100:.2f}%)")
print()

# ========================
# PHASE 2: FINE-TUNE
# ========================
print("STEP 4: PHASE 2 - Fine-Tuning (30 epochs)...")
print("-" * 70)
print("Unfreezing top layers of base model...")

base_model.trainable = True

# Freeze first 200 layers
for layer in base_model.layers[:200]:
    layer.trainable = False

print(f"✓ Trainable layers: {sum([layer.trainable for layer in model.layers])}")

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE / 5, weight_decay=0.0001),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc')
    ]
)

callbacks_phase2 = [
    ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, 'best_model_phase2.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=4,
        min_lr=1e-9,
        verbose=1
    )
]

history_phase2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=30,
    callbacks=callbacks_phase2,
    verbose=1
)

print()
print("✓ Phase 2 completed!")
val_acc_phase2 = max(history_phase2.history['val_accuracy'])
print(f"✓ Best validation accuracy: {val_acc_phase2:.4f} ({val_acc_phase2*100:.2f}%)")
print()

# ========================
# EVALUATE
# ========================
print("STEP 5: Final Evaluation on Test Set...")
print("-" * 70)

test_results = model.evaluate(test_generator, verbose=1)
test_loss = test_results[0]
test_accuracy = test_results[1]
test_precision = test_results[2]
test_recall = test_results[3]
test_auc = test_results[4]

print()
print("=" * 70)
print("FINAL TEST RESULTS:")
print("=" * 70)
print(f"  • Loss:      {test_loss:.4f}")
print(f"  • Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Precision: {test_precision:.4f} ({test_precision*100:.2f}%)")
print(f"  • Recall:    {test_recall:.4f} ({test_recall*100:.2f}%)")
print(f"  • AUC:       {test_auc:.4f}")
print()

# Per-class evaluation
print("PER-CLASS PERFORMANCE:")
print("-" * 70)

test_generator.reset()
y_pred = model.predict(test_generator, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

report = classification_report(
    y_true,
    y_pred_classes,
    target_names=CLASSES,
    digits=4
)
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASSES,
    yticklabels=CLASSES,
    cbar_kws={'label': 'Count'}
)
plt.title('Confusion Matrix - Best Model', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix_best.png', dpi=300)
print("✓ Confusion matrix saved")
print()

# ========================
# SAVE MODEL
# ========================
final_model_path = os.path.join(MODEL_SAVE_PATH, 'brain_tumor_model_BEST.h5')
model.save(final_model_path)
print(f"✓ Final model saved to: {final_model_path}")
print()

# ========================
# PLOT TRAINING HISTORY
# ========================
combined_acc = history_phase1.history['accuracy'] + history_phase2.history['accuracy']
combined_val_acc = history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy']
combined_loss = history_phase1.history['loss'] + history_phase2.history['loss']
combined_val_loss = history_phase1.history['val_loss'] + history_phase2.history['val_loss']

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(combined_acc, label='Training', linewidth=2, color='#0066CC')
ax1.plot(combined_val_acc, label='Validation', linewidth=2, color='#00A3E0')
ax1.axvline(x=30, color='red', linestyle='--', label='Fine-tuning', alpha=0.7)
ax1.set_title('Model Accuracy - Best Training', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(combined_loss, label='Training', linewidth=2, color='#0066CC')
ax2.plot(combined_val_loss, label='Validation', linewidth=2, color='#00A3E0')
ax2.axvline(x=30, color='red', linestyle='--', label='Fine-tuning', alpha=0.7)
ax2.set_title('Model Loss - Best Training', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('training_history_best.png', dpi=150)
print("✓ Training history saved")
print()

# ========================
# SUMMARY
# ========================
print("=" * 70)
print("✅ BEST MODEL TRAINING COMPLETE!")
print("=" * 70)
print()

if test_accuracy >= 0.90:
    print("🎉 EXCELLENT! Accuracy >= 90%")
    print("   This model is production-ready!")
elif test_accuracy >= 0.85:
    print("✓ VERY GOOD! Accuracy >= 85%")
    print("   Model performs well!")
else:
    print("⚠ Accuracy lower than expected")
    print("   Consider training longer or checking data")

print()
print("To use this model:")
print(f"1. Update api_server.py to load: {final_model_path}")
print("2. Restart backend server")
print("3. Test with frontend!")
print()
print("=" * 70)