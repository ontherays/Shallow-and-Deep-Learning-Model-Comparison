import numpy as np

# Flatten the images
X_train = X_train.reshape((60000, 784)).astype("float32") / 255.0
X_test = X_test.reshape((10000, 784)).astype("float32") / 255.0

X_train.shape, X_test.shape
