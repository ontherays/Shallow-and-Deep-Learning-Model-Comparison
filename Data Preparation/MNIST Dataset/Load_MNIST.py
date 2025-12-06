from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalize grayscale to 0–1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

# Create validation set: 50k train / 10k val
X_train, X_val = X_train[:50000], X_train[50000:]
y_train, y_val = y_train[:50000], y_train[50000:]

# Add channel dimension (required by Conv2D)
X_train = np.expand_dims(X_train, -1)
X_val = np.expand_dims(X_val, -1)
X_test = np.expand_dims(X_test, -1)

X_train.shape, X_val.shape, X_test.shape
