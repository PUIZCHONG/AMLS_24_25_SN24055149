import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, learning_curve
import matplotlib.pyplot as plt
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. Load data
data_dir = r'C:\Users\SIMON\Desktop\ELEC0134-AMLS\bloodmnist.npz'
data = np.load(data_dir)
X_train = data['train_images']  # shape: (11959, 28, 28, 1, 3)?  We'll fix that below
y_train = data['train_labels']
X_val   = data['val_images']
y_val   = data['val_labels']
X_test  = data['test_images']
y_test  = data['test_labels']

# 2. Fix shape from (N, 28, 28, 1, 3) to (N, 28, 28, 3)
X_train = np.squeeze(X_train)  # => (11959, 28, 28, 3)
X_val   = np.squeeze(X_val)    # => (1715, 28, 28, 3)
X_test  = np.squeeze(X_test)   # => (3421, 28, 28, 3)

# If you truly want GRAYSCALE:
X_train = X_train.mean(axis=-1, keepdims=True)  # => (11959, 28, 28, 1)
X_val   = X_val.mean(axis=-1, keepdims=True)    # => (1715, 28, 28, 1)
X_test  = X_test.mean(axis=-1, keepdims=True)   # => (3421, 28, 28, 1)

# Normalize
X_train = X_train.astype("float32") / 255.0
X_val   = X_val.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

y_train = y_train.ravel()
y_val   = y_val.ravel()
y_test  = y_test.ravel()

print("After fix:")
print("X_train shape:", X_train.shape)  # (11959, 28, 28, 1)
print("X_val shape:", X_val.shape)      # (1715, 28, 28, 1)
print("X_test shape:", X_test.shape)    # (3421, 28, 28, 1)

# 3. Data augmentation
data_augmentation = keras.Sequential([
     layers.RandomFlip("horizontal", seed=SEED),
    layers.RandomRotation(0.1, seed=SEED),
    layers.RandomZoom(0.1, seed=SEED),
], name='data_augmentation')

batch_size = 64
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .batch(batch_size)
            .map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .batch(batch_size)
          .prefetch(tf.data.AUTOTUNE))

test_ds = (tf.data.Dataset.from_tensor_slices((X_test, y_test))
           .batch(batch_size)
           .prefetch(tf.data.AUTOTUNE))

# 4. Define a small ResNet-like model (ResNet-28) for grayscale
class BasicBlock(layers.Layer):
    def __init__(self, filters, strides=1):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False)
        self.bn1   = layers.BatchNormalization()
        self.relu  = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False)
        self.bn2   = layers.BatchNormalization()

        self.shortcut = None
        if strides != 1:
            self.shortcut = keras.Sequential([
                layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                layers.BatchNormalization()
            ])

    def call(self, x, training=False):
        shortcut = x
        x = self.conv1(x, training=training)
        x = self.bn1(x, training=training)
        x = self.relu(x)
        x = self.conv2(x, training=training)
        x = self.bn2(x, training=training)

        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut, training=training)

        x = layers.add([x, shortcut])
        x = self.relu(x)
        return x

def build_resnet28(input_shape=(28, 28, 1), num_classes=8):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, strides=1, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = BasicBlock(16, 1)(x)
    x = BasicBlock(16, 1)(x)
    x = BasicBlock(32, 2)(x)
    x = BasicBlock(32, 1)(x)
    x = BasicBlock(64, 2)(x)
    x = BasicBlock(64, 1)(x)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs, outputs, name="ResNet28")

model = build_resnet28()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# Train
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=64,  # can try 32, 64, etc.
    callbacks=callbacks,
    shuffle=True
)
# Evaluate
test_loss, test_acc = model.evaluate(test_ds)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# Feature extraction for SVM
feature_extractor = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)

def get_features_and_labels(x, y):
    feats, labs = [], []
    ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
    for xb, yb in ds:
        f = feature_extractor(xb, training=False)
        feats.append(f.numpy())
        labs.append(yb.numpy())
    return np.concatenate(feats, axis=0), np.concatenate(labs, axis=0)

train_features, train_labels = get_features_and_labels(X_train, y_train)
val_features, val_labels = get_features_and_labels(X_val, y_val)
test_features, test_labels = get_features_and_labels(X_test, y_test)

svm = SVC(kernel='rbf', probability=True)
svm.fit(train_features, train_labels)

val_pred = svm.predict(val_features)
print("SVM (val) classification report:")
print(classification_report(val_labels, val_pred))

test_pred = svm.predict(test_features)
print("SVM (test) classification report:")
print(classification_report(test_labels, test_pred))


val_acc_svm = search.best_estimator_.score(val_features, y_val)
test_acc_svm = search.best_estimator_.score(test_features, y_test)
print("SVM Validation Accuracy:", val_acc_svm)
print("SVM Test Accuracy:", test_acc_svm)