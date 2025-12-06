import tensorflow as tf
import numpy as np

vanish_model = tf.keras.Sequential()
for _ in range(10):
    vanish_model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
vanish_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

x_batch = X_train[:256]
y_batch = y_train[:256]

with tf.GradientTape() as tape:
    preds = vanish_model(x_batch)
    loss = tf.keras.losses.binary_crossentropy(y_batch, preds)

grads = tape.gradient(loss, vanish_model.trainable_weights)

print("Gradient norms per layer:")
for i in range(0, len(grads), 2):
    print(f"Layer {i//2}: {tf.norm(grads[i]).numpy()}")
