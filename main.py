import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
# -------------------------------------
# Ensure deterministic operations
# -------------------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# ---------------------------
# Load the BreastMNIST dataset
# ---------------------------
data_dir = r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\BreastMNIST.npz'
data = np.load(data_dir)
x_train = data['train_images']  # shape: (546, 28, 28)
y_train = data['train_labels']  # shape: (546, )
x_val = data['val_images']      # shape: (78, 28, 28)
y_val = data['val_labels']
x_test = data['test_images']    # shape: (156, 28, 28)
y_test = data['test_labels']

# Convert images to float32 and scale to [0,1]
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# Add a channel dimension (grayscale -> single channel)
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# Ensure labels are 1D arrays for scikit-learn
y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()
# --------------------------
# Data Augmentation Pipeline
# --------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal", seed=42),
    layers.RandomRotation(0.1, seed=42),
    layers.RandomZoom(0.1, seed=42),
    layers.RandomContrast(0.1, seed=42),
], name="data_augmentation")

# ---------------------------
# Build a Small CNN Model
# ---------------------------
def create_cnn_model():
    inputs = keras.Input(shape=(28, 28, 1))
    x = data_augmentation(inputs)  # Apply augmentation on-the-fly
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_cnn_model()

# ---------------------------
# Train the CNN Model
# ---------------------------
early_stopping = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True,verbose=1)


history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=45,
    batch_size=32,
    callbacks=[early_stopping],
    shuffle=True
)


# Extract accuracy and loss from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_acc) + 1)

# Plot Accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.title('CNN Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1,2,2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('CNN Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
# ---------------------------
# Extract Features using CNN
# ---------------------------
feature_extractor = keras.Model(inputs=model.input, outputs=model.layers[-2].output)  # second last layer outputs

train_features = feature_extractor.predict(x_train)
val_features = feature_extractor.predict(x_val)
test_features = feature_extractor.predict(x_test)

# Standardize features before SVM training
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# ---------------------------
# Train an SVM Classifier
# ---------------------------
# Note: scikit-learn doesn't have a native 'relu' kernel. 
# You could use a custom kernel function, or try 'rbf' as a starting point.
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']  # Start with standard kernels
}
def relu_kernel(X, Y):
    # A custom kernel example (X and Y are feature matrices)
    # Compute dot product:
    dot_product = np.dot(X, Y.T)
    # Apply ReLU:
    return np.maximum(dot_product, 0)

svm = SVC(kernel=relu_kernel)  # Using our custom kernel
# Initialize GridSearchCV
grid_search = GridSearchCV(
    svm, 
    param_grid, 
    scoring='accuracy', 
    cv=3,
    n_jobs=-1,
    verbose=2
)

# Fit GridSearchCV
grid_search.fit(train_features, y_train)

# Get the best estimator
best_svm = grid_search.best_estimator_
print("Best SVM Parameters:", grid_search.best_params_)

# Evaluate on validation set
val_acc = best_svm.score(val_features, y_val)
print("Validation Accuracy:", val_acc)

# Evaluate on test set
test_acc = best_svm.score(test_features, y_test)
print("Test Accuracy:", test_acc)

# ---------------------------
# Plot SVM Learning Curves
# ---------------------------
def plot_svm_learning_curve(estimator, X, y, cv=3, train_sizes=np.linspace(0.1, 1.0, 5)):
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes, scoring='accuracy', random_state=42
    )

    train_means = np.mean(train_scores, axis=1)
    train_stds  = np.std(train_scores, axis=1)
    valid_means = np.mean(valid_scores, axis=1)
    valid_stds  = np.std(valid_scores, axis=1)

    plt.figure(figsize=(8,6))
    plt.title(f"SVM Learning Curve ({estimator.kernel} kernel)")
    plt.plot(train_sizes, train_means, 'o-', color="r", label="Training score")
    plt.fill_between(train_sizes, train_means - train_stds, train_means + train_stds, alpha=0.1, color="r")
    plt.plot(train_sizes, valid_means, 'o-', color="g", label="Cross-validation score")
    plt.fill_between(train_sizes, valid_means - valid_stds, valid_means + valid_stds, alpha=0.1, color="g")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

# Plot learning curves for the best SVM
plot_svm_learning_curve(best_svm, train_features, y_train)

# Optional: If you want to compare different kernels, you can loop through them
kernels = ['linear', 'rbf', 'poly', 'sigmoid']
for kernel in kernels:
    svm = SVC(kernel=kernel, C=grid_search.best_params_['C'])  # Use best C found
    plot_svm_learning_curve(svm, train_features, y_train)