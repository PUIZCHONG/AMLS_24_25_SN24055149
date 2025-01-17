import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

# -------------------------------------------
# 1. Ensure Deterministic Operations (Optional)
# -------------------------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------------------------
# 2. Load Your Dataset
# -------------------------------------------
data_dir = r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\bloodmnist.npz'
data = np.load(data_dir)
x_train = data['train_images']   # shape: (11959, 28, 28)
y_train = data['train_labels']   # shape: (11959,)
x_val   = data['val_images']     # shape: (1715, 28, 28)
y_val   = data['val_labels']
x_test  = data['test_images']    # shape: (3421, 28, 28)
y_test  = data['test_labels']

# -------------------------------------------
# 3. Preprocess Images (Normalize + Reshape)
# -------------------------------------------
# Normalize to [0,1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Convert labels to int
y_train = y_train.astype('int')
y_val   = y_val.astype('int')
y_test  = y_test.astype('int')

# -------------------------------------------
# 4. Enhanced Data Augmentation
# -------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, fill_mode='nearest', seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
], name="data_augmentation")

# -------------------------------------------
# 5. Build CNN (with Dropout & L2 Regularization)
# -------------------------------------------
def create_enhanced_cnn():
    l2_reg = regularizers.l2(1e-4)  # L2 regularization factor

    inputs = keras.Input(shape=(28, 28, 3))  # Corrected to 1 channel
    x = data_augmentation(inputs)

    # Conv block 1
    x = layers.Conv2D(
        32, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2_reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D()(x)

    # Conv block 2
    x = layers.Conv2D(
        64, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2_reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D()(x)

    # Conv block 3
    x = layers.Conv2D(
        128, (3, 3), activation='relu', padding='same',
        kernel_regularizer=l2_reg
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.MaxPooling2D()(x)

    # Flatten & Dense
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        128, activation='relu',
        kernel_regularizer=l2_reg
    )(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(8, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_enhanced_cnn()
model.summary()

# -------------------------------------------
# 6. Select a 10,000-Sample Subset from Training Data
# -------------------------------------------
# Define the desired subset size
subset_size = 10000

# Check if subset_size is valid
total_train_samples = x_train.shape[0]
if subset_size > total_train_samples:
    raise ValueError(f"subset_size {subset_size} exceeds total training samples {total_train_samples}")

# Randomly select 10,000 samples using train_test_split
x_train_subset, _, y_train_subset, _ = train_test_split(
    x_train,
    y_train,
    train_size=subset_size,
    random_state=SEED,
    shuffle=True
)

print(f"Training on a subset of {subset_size} samples out of {total_train_samples}.")

# -------------------------------------------
# 7. Train the CNN with the 10,000-Sample Subset
# -------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

history = model.fit(
    x_train_subset, y_train_subset,  # Use the subset here
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=callbacks,
    shuffle=True
)

# -------------------------------------------
# 8. Plot Training Curves
# -------------------------------------------
def plot_history(h):
    train_acc = h.history['accuracy']
    val_acc   = h.history['val_accuracy']
    train_loss = h.history['loss']
    val_loss   = h.history['val_loss']

    epochs_range = range(1, len(train_acc) + 1)

    plt.figure(figsize=(10,4))
    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_acc, 'b-o', label='Train Acc')
    plt.plot(epochs_range, val_acc, 'r-o', label='Val Acc')
    plt.title('CNN Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_loss, 'b-o', label='Train Loss')
    plt.plot(epochs_range, val_loss, 'r-o', label='Val Loss')
    plt.title('CNN Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# -------------------------------------------
# 9. Evaluate CNN on Test Set
# -------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"[CNN] Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# -------------------------------------------
# 10. Plot Confusion Matrix
# -------------------------------------------
# Predict class probabilities
y_pred_probs = model.predict(x_test, batch_size=256, verbose=1)

# Convert probabilities to class labels
y_pred = np.argmax(y_pred_probs, axis=1)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Define class names (replace with actual class names if available)
class_names = [f"Class {i}" for i in range(8)]

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# (Optional) Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=class_names))

# -------------------------------------------
# 11. Plot Learning Curve (Optional)
# -------------------------------------------
def compute_learning_curve(model_builder, x_train, y_train, x_val, y_val, train_sizes, epochs=50, batch_size=64, callbacks=None):
    train_scores = []
    val_scores = []

    for size in train_sizes:
        print(f"Training with {size} samples...")
        # Select a subset of the training data
        indices = np.random.choice(len(x_train), size=size, replace=False)
        x_subset = x_train[indices]
        y_subset = y_train[indices]

        # Build a new model instance
        model = model_builder()  # No additional arguments

        # Train the model
        history = model.fit(
            x_subset, y_subset,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0  # Suppress training output for clarity
        )

        # Evaluate the model
        train_loss, train_acc = model.evaluate(x_subset, y_subset, verbose=0)
        val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)

        train_scores.append(train_acc)
        val_scores.append(val_acc)

    return train_scores, val_scores

# Define training sizes (up to 10,000)
train_sizes = [1000, 3000, 5000, 8000, 10000]

# Compute learning curve data
learning_callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=0)
]

train_scores, val_scores = compute_learning_curve(
    create_enhanced_cnn,
    x_train_subset, y_train_subset,  # Use the subset for learning curve
    x_val, y_val,
    train_sizes=train_sizes,
    epochs=50,
    batch_size=64,
    callbacks=learning_callbacks
)

# Plot the learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores, 'o-', color='blue', label='Training Accuracy')
plt.plot(train_sizes, val_scores, 'o-', color='red', label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------
# 12. CNN Feature Extraction + SVM
# -------------------------------------------
# Extract features from the second-to-last layer (Dense(128))
feature_extractor = keras.Model(
    inputs=model.input,
    outputs=model.layers[-2].output  # "Dense(128)"
)

# Extract features
print("Extracting features using CNN...")
train_features = feature_extractor.predict(x_train_subset, batch_size=256, verbose=1)
val_features   = feature_extractor.predict(x_val, batch_size=256, verbose=1)
test_features  = feature_extractor.predict(x_test, batch_size=256, verbose=1)

# Scale features (important for SVM)
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features   = scaler.transform(val_features)
test_features  = scaler.transform(test_features)

# Train SVM
print("Training SVM...")
svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=SEED)
svm.fit(train_features, y_train_subset)

# Evaluate on val/test sets
val_preds_svm = svm.predict(val_features)
val_acc_svm = accuracy_score(y_val, val_preds_svm)

test_preds_svm = svm.predict(test_features)
test_acc_svm = accuracy_score(y_test, test_preds_svm)

print(f"[SVM] Validation Accuracy: {val_acc_svm:.4f}")
print(f"[SVM] Test Accuracy:      {test_acc_svm:.4f}")

# -------------------------------------------
# 13. Ensemble Method: Combine CNN & SVM
# -------------------------------------------
# Simple weighted average ensemble

# 13a. CNN probabilities (softmax)
cnn_probs_val  = model.predict(x_val, batch_size=256, verbose=1)   # shape: (val_samples, 8)
cnn_probs_test = model.predict(x_test, batch_size=256, verbose=1)  # shape: (test_samples, 8)

# 13b. SVM probabilities
svm_probs_val  = svm.predict_proba(val_features)   # shape: (val_samples, 8)
svm_probs_test = svm.predict_proba(test_features)  # shape: (test_samples, 8)

# 13c. Weighted average ensemble. Adjust weights as desired (e.g., 0.5 CNN, 0.5 SVM)
ensemble_probs_val  = 0.5 * cnn_probs_val  + 0.5 * svm_probs_val
ensemble_probs_test = 0.5 * cnn_probs_test + 0.5 * svm_probs_test

# 13d. Convert probabilities to class predictions
ensemble_preds_val  = np.argmax(ensemble_probs_val, axis=1)
ensemble_preds_test = np.argmax(ensemble_probs_test, axis=1)

# 13e. Evaluate ensemble accuracy
ensemble_val_acc  = accuracy_score(y_val,  ensemble_preds_val)
ensemble_test_acc = accuracy_score(y_test, ensemble_preds_test)

print(f"[Ensemble] Validation Accuracy: {ensemble_val_acc:.4f}")
print(f"[Ensemble] Test Accuracy:       {ensemble_test_acc:.4f}")
