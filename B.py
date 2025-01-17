import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,  regularizers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt

# -----------------------------
# 1. Fix All Random Seeds
# -----------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 2. Load BloodMNIST Data
# -----------------------------
# This is just an example. Adapt to your dataset path.
data_dir= r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\bloodmnist.npz'
data = np.load(data_dir)
x_train = data['train_images']  # shape: (11959, 28, 28)
y_train = data['train_labels']  # shape: (11959,)
x_val   = data['val_images']    # shape: (1715, 28, 28)
y_val   = data['val_labels']
x_test  = data['test_images']   # shape: (3421, 28, 28)
y_test  = data['test_labels']

# -----------------------------
# 3. Preprocessing
# -----------------------------
# Normalize pixel values to [0,1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Convert labels to int
y_train = y_train.astype('int')
y_val = y_val.astype('int')
y_test = y_test.astype('int')

# One-hot encode if you plan to use tf.keras losses for multiclass
# But for sparse_categorical_crossentropy, you can keep them as integers
y_train_cat = keras.utils.to_categorical(y_train, num_classes=8)
y_val_cat = keras.utils.to_categorical(y_val, num_classes=8)
y_test_cat = keras.utils.to_categorical(y_test, num_classes=8)

# -----------------------------
# 4. Data Augmentation (Optional)
# -----------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
], name="data_augmentation")

# -----------------------------
# 5. Build a CNN Model
# -----------------------------
def build_cnn_model(num_conv_layers=3):
    """
    Example CNN builder where you can choose how many conv layers to stack.
    """
    inputs = keras.Input(shape=(28, 28, 3))
    
    # Apply augmentation if desired
    x = data_augmentation(inputs)   

    # Example: stack 'num_conv_layers' conv+maxpool blocks
    filters = 32
    for i in range(num_conv_layers):
        x = layers.Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D()(x)
        x = layers.BatchNormalization()(x)
        filters *= 2  # double the filters each block, for instance

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(8, activation='softmax')(x)  # 8 classes

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',  # using integer labels
        metrics=['accuracy']
    )
    return model

# You can experiment with different numbers of conv layers, e.g., 3 or 5
model = build_cnn_model(num_conv_layers=3)

# -----------------------------
# 6. Train the CNN Model
# -----------------------------
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=50,
    batch_size=64,  # can try 32, 64, etc.
    callbacks=callbacks,
    shuffle=True
)

# -----------------------------
# 7. Plot CNN Training Curves
# -----------------------------
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
plt.title('CNN Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('CNN Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# -----------------------------
# 8. Evaluate on Test Set
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Test Accuracy:", test_acc)

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

search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
search.fit(train_features, y_train)  # still using integer labels
print("Best SVM params:", search.best_params_)

val_acc_svm = search.best_estimator_.score(val_features, y_val)
test_acc_svm = search.best_estimator_.score(test_features, y_test)
print("SVM Validation Accuracy:", val_acc_svm)
print("SVM Test Accuracy:", test_acc_svm)