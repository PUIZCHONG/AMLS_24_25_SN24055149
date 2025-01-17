import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt
# -------------------------------------
# 1. Ensure Deterministic Operations
# -------------------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# 2. Load BloodMNIST dataset
# ---------------------------

data_dir= r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\bloodmnist.npz'
data = np.load(data_dir)
x_train = data['train_images']  # shape: (11959, 28, 28)
y_train = data['train_labels']  # shape: (11959,)
x_val   = data['val_images']    # shape: (1715, 28, 28)
y_val   = data['val_labels']
x_test  = data['test_images']   # shape: (3421, 28, 28)
y_test  = data['test_labels']

# ------------------------------------------------
# 3. Preprocessing (Normalization + Reshaping)
# ------------------------------------------------
# Normalize to [0,1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# If images are truly grayscale, shape is (batch, 28, 28)
# Expand dims to (batch, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)  # => (11959, 28, 28, 1)
x_val   = np.expand_dims(x_val,   axis=-1)  # => (1715,  28, 28, 1)
x_test  = np.expand_dims(x_test,  axis=-1)  # => (3421,  28, 28, 1)

# Convert labels to integer (if not already)
y_train = y_train.astype('int')
y_val   = y_val.astype('int')
y_test  = y_test.astype('int')

# ------------------------------------------------
# 4. Data Augmentation (Optional but recommended)
# ------------------------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
], name="data_augmentation")

# ------------------------------------------------
# 5. Define a 5-Layer CNN Model
# ------------------------------------------------
def build_cnn_model_5layers():
    """
    Builds a CNN with 5 convolutional layers, each followed by MaxPooling and BatchNorm,
    then a fully connected head. Output: 8-class classification for BloodMNIST.
    """
    inputs = keras.Input(shape=(28, 28, 3))

    # Data Augmentation
    x = data_augmentation(inputs)

    # Layer Block 1
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Layer Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Layer Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Layer Block 4
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.BatchNormalization()(x)

    # Layer Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Flatten + Dense layers
    x = layers.Flatten()(x)
    x = layers.Dropout(0.4)(x)    # Increase dropout with deeper networks
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # 8 classes for BloodMNIST
    outputs = layers.Dense(8, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_cnn_model_5layers()
model.summary()  # Optional: print the model architecture

# ------------------------------------------------
# 6. Train the Model
# ------------------------------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,           # Try up to 50, early stopping may halt sooner
    batch_size=64,       # Adjust batch size as needed
    callbacks=callbacks,
    shuffle=True,        # With seeds set, this is still deterministic
    verbose=1
)

# ------------------------------------------------
# 7. Plot Accuracy & Loss
# ------------------------------------------------
train_acc = history.history['accuracy']
val_acc   = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss   = history.history['val_loss']

epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(10,4))
# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, train_acc, 'bo-', label='Training Acc')
plt.plot(epochs_range, val_acc, 'ro-', label='Validation Acc')
plt.title('5-Layer CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('5-Layer CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# ------------------------------------------------
# 8. Evaluate on the Test Set
# ------------------------------------------------
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
# -----------------------------
# (Optional) 9. CNN Feature Extraction + SVM
# -----------------------------
# If you want to see how SVM kernels do on top of CNN features:
feature_extractor = keras.Model(
    inputs=model.input, 
    outputs=model.layers[-2].output  # second-last Dense layer (128-d)
)

train_features = feature_extractor.predict(x_train)
val_features   = feature_extractor.predict(x_val)
test_features  = feature_extractor.predict(x_test)

# Scale the features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features   = scaler.transform(val_features)
test_features  = scaler.transform(test_features)

# Multi-class SVM
svm = SVC()  # default is rbf kernel
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
search.fit(train_features, y_train)  # still using integer labels
print("Best SVM params:", search.best_params_)

val_acc_svm = search.best_estimator_.score(val_features, y_val)
test_acc_svm = search.best_estimator_.score(test_features, y_test)
print("SVM Validation Accuracy:", val_acc_svm)
print("SVM Test Accuracy:", test_acc_svm)