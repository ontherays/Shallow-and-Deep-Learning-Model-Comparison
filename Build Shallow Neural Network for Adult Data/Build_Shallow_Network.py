import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Shallow neural network (2 hidden layers)
shallow_model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

shallow_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

shallow_model.summary()
