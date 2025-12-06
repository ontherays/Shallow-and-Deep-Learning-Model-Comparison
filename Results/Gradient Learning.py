import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Deep model with sigmoid activation
def build_vanishing_model(input_dim):
    model = tf.keras.Sequential()
    for _ in range(10):
        model.add(tf.keras.layers.Dense(64, activation="sigmoid"))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    return model

model_vanish = build_vanishing_model(X_train.shape[1])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_fn = tf.keras.losses.BinaryCrossentropy()

batch_size = 256
gradient_history = {i: [] for i in range(len(model_vanish.layers))}

# Custom training loop for minibatch gradient extraction
for epoch in range(3):
    print(f"\nEPOCH {epoch+1}")
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)

    for step in range(0, len(X_train), batch_size):
        batch_idx = indices[step:step + batch_size]
        x_batch = X_train[batch_idx]
        y_batch = y_train[batch_idx]

        with tf.GradientTape() as tape:
            preds = model_vanish(x_batch, training=True)
            loss = loss_fn(y_batch, preds)

        grads = tape.gradient(loss, model_vanish.trainable_weights)

        # Track gradient norms per layer
        w_idx = 0
        for layer_i, layer in enumerate(model_vanish.layers):
            layer_grads = grads[w_idx]
            w_idx += 2 if isinstance(layer, tf.keras.layers.Dense) else 0

            if layer_grads is not None:
                gradient_norm = tf.norm(layer_grads).numpy()
                gradient_history[layer_i].append(gradient_norm)

    print(f"Epoch {epoch+1} finished")

# Plot gradient norms of each layer
plt.figure(figsize=(12,6))
for layer_i, grads in gradient_history.items():
    plt.plot(grads, label=f"Layer {layer_i}")

plt.yscale("log")
plt.title("Gradient Norms per Layer (Log Scale)")
plt.xlabel("Training Step")
plt.ylabel("Gradient Norm")
plt.legend()
plt.show()
